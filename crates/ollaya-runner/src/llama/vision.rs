//! Still-image support through libmtmd from the same pinned release as libllama.
//! All operations run under LlamaModel's context mutex. PNG decoding uses Ollaya's
//! existing decoder; no URLs, paths, video subprocesses or audio are accepted.
use super::model_error;
use crate::{Error, vision};
use libloading::Library;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::Path;
use std::ptr::NonNull;

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    use_gpu: bool,
    device: *mut c_void,
    print_timings: bool,
    n_threads: c_int,
    image_marker: *const c_char,
    media_marker: *const c_char,
    flash_attn_type: c_int,
    warmup: bool,
    image_min_tokens: c_int,
    image_max_tokens: c_int,
    cb_eval: *const c_void,
    cb_eval_user_data: *mut c_void,
    batch_max_tokens: i32,
    progress_callback: *const c_void,
    progress_callback_user_data: *mut c_void,
}
#[repr(C)]
struct Text {
    text: *const c_char,
    text_len: usize,
    add_special: bool,
    parse_special: bool,
}
struct Api {
    _lib: Library,
    defaults: unsafe extern "C" fn() -> Params,
    init: unsafe extern "C" fn(*const c_char, *const c_void, Params) -> *mut c_void,
    free: unsafe extern "C" fn(*mut c_void),
    marker: unsafe extern "C" fn(*const c_void) -> *const c_char,
    supports_vision: unsafe extern "C" fn(*const c_void) -> bool,
    bitmap: unsafe extern "C" fn(u32, u32, *const u8) -> *mut c_void,
    bitmap_free: unsafe extern "C" fn(*mut c_void),
    chunks: unsafe extern "C" fn() -> *mut c_void,
    chunks_free: unsafe extern "C" fn(*mut c_void),
    tokenize: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        *const Text,
        *const *const c_void,
        usize,
    ) -> i32,
    positions: unsafe extern "C" fn(*const c_void) -> i32,
    count: unsafe extern "C" fn(*const c_void) -> usize,
    chunk: unsafe extern "C" fn(*const c_void, usize) -> *const c_void,
    chunk_type: unsafe extern "C" fn(*const c_void) -> c_int,
    chunk_tokens: unsafe extern "C" fn(*const c_void) -> usize,
    non_causal: unsafe extern "C" fn(*const c_void, *const c_void) -> bool,
    eval: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        *const c_void,
        i32,
        i32,
        i32,
        bool,
        *mut i32,
    ) -> i32,
}
impl Api {
    fn load(dir: &Path) -> Result<Self, Error> {
        let name = if cfg!(target_os = "windows") {
            "mtmd.dll"
        } else if cfg!(target_os = "macos") {
            "libmtmd.0.dylib"
        } else {
            "libmtmd.so.0"
        };
        // SAFETY: these signatures mirror tools/mtmd/{mtmd,mtmd-helper}.h at
        // 7fe450e19305b828c199d602c23a8337aaa1f03b. Library lives as long as its symbols.
        unsafe {
            let lib = Library::new(dir.join(name)).map_err(|e| {
                model_error(format!(
                    "load {name}: {e}; reinstall llama.cpp libraries with multimodal support"
                ))
            })?;
            macro_rules! symbol {
                ($name:literal) => {
                    *lib.get(concat!($name, "\0").as_bytes())
                        .map_err(|e| model_error(format!("libmtmd: {e}")))?
                };
            }
            Ok(Self {
                defaults: symbol!("mtmd_context_params_default"),
                init: symbol!("mtmd_init_from_file"),
                free: symbol!("mtmd_free"),
                marker: symbol!("mtmd_get_marker"),
                supports_vision: symbol!("mtmd_support_vision"),
                bitmap: symbol!("mtmd_bitmap_init"),
                bitmap_free: symbol!("mtmd_bitmap_free"),
                chunks: symbol!("mtmd_input_chunks_init"),
                chunks_free: symbol!("mtmd_input_chunks_free"),
                tokenize: symbol!("mtmd_tokenize"),
                positions: symbol!("mtmd_helper_get_n_pos"),
                count: symbol!("mtmd_input_chunks_size"),
                chunk: symbol!("mtmd_input_chunks_get"),
                chunk_type: symbol!("mtmd_input_chunk_get_type"),
                chunk_tokens: symbol!("mtmd_input_chunk_get_n_tokens"),
                non_causal: symbol!("mtmd_decode_use_non_causal"),
                eval: symbol!("mtmd_helper_eval_chunks"),
                _lib: lib,
            })
        }
    }
}
pub(super) struct Vision {
    api: Api,
    ptr: NonNull<c_void>,
}
impl Drop for Vision {
    fn drop(&mut self) {
        // SAFETY: initialized once, freed before the text model it borrows.
        unsafe {
            (self.api.free)(self.ptr.as_ptr());
        }
    }
}
impl Vision {
    pub(super) fn load(
        dir: &Path,
        projector: &Path,
        model: *const c_void,
        gpu: bool,
        threads: i32,
    ) -> Result<Self, Error> {
        let api = Api::load(dir)?;
        let path = super::cpath(projector).map_err(Error::Model)?;
        // SAFETY: model and path outlive initialization; returned handle owned here.
        let ptr = unsafe {
            let mut params = (api.defaults)();
            params.use_gpu = gpu;
            params.n_threads = threads;
            params.warmup = false;
            NonNull::new((api.init)(path.as_ptr(), model, params))
                .ok_or_else(|| model_error("could not load matching vision projector"))?
        };
        let vision = Self { api, ptr };
        // SAFETY: valid projector handle.
        if !unsafe { (vision.api.supports_vision)(vision.ptr.as_ptr()) } {
            return Err(model_error("projector does not support vision"));
        }
        Ok(vision)
    }
    pub(super) fn prefix(&self, text: &str, images: &[Vec<u8>]) -> Result<Chunks<'_>, Error> {
        if images.len() > 16 {
            return Err(ollaya_decision::Error::invalid("Winnow accepts at most 16 images").into());
        }
        let mut bitmaps = Vec::new();
        for bytes in images {
            let image = vision::decode(bytes)?;
            // SAFETY: RGB buffer is width*height*3 bytes; mtmd copies it.
            let ptr = unsafe {
                (self.api.bitmap)(image.width as u32, image.height as u32, image.data.as_ptr())
            };
            bitmaps.push(Bitmap {
                api: &self.api,
                ptr: NonNull::new(ptr)
                    .ok_or_else(|| model_error("mtmd bitmap allocation failed"))?,
            });
        }
        // SAFETY: marker belongs to the live projector and is NUL terminated.
        let marker = unsafe { CStr::from_ptr((self.api.marker)(self.ptr.as_ptr())) }
            .to_str()
            .map_err(|e| model_error(e.to_string()))?;
        let marked = text.replace("<__media__>", marker);
        let text = CString::new(marked).map_err(|e| model_error(e.to_string()))?;
        let input = Text {
            text: text.as_ptr(),
            text_len: text.as_bytes().len(),
            add_special: true,
            parse_special: true,
        };
        let pointers: Vec<*const c_void> = bitmaps
            .iter()
            .map(|b| b.ptr.as_ptr().cast_const())
            .collect();
        // SAFETY: chunks and bitmap handles remain live through tokenization/evaluation.
        let ptr = unsafe {
            NonNull::new((self.api.chunks)())
                .ok_or_else(|| model_error("mtmd chunks allocation failed"))?
        };
        let chunks = Chunks {
            vision: self,
            ptr,
            _bitmaps: bitmaps,
        };
        let rc = unsafe {
            (self.api.tokenize)(
                self.ptr.as_ptr(),
                ptr.as_ptr(),
                &input,
                pointers.as_ptr(),
                pointers.len(),
            )
        };
        if rc != 0 {
            return Err(model_error(format!("image tokenization failed ({rc})")));
        }
        // SAFETY: each chunk belongs to this successfully tokenized list. Non-causal
        // image attention must fit one physical microbatch, as in the author's server.
        unsafe {
            for i in 0..(self.api.count)(ptr.as_ptr()) {
                let chunk = (self.api.chunk)(ptr.as_ptr(), i);
                if (self.api.chunk_type)(chunk) == 1
                    && (self.api.non_causal)(self.ptr.as_ptr(), chunk)
                    && (self.api.chunk_tokens)(chunk) > super::N_UBATCH as usize
                {
                    return Err(ollaya_decision::Error::invalid(
                        "image exceeds the vision microbatch; resize the image",
                    )
                    .into());
                }
            }
        }
        Ok(chunks)
    }
}
struct Bitmap<'a> {
    api: &'a Api,
    ptr: NonNull<c_void>,
}
impl Drop for Bitmap<'_> {
    fn drop(&mut self) {
        // SAFETY: uniquely owned bitmap.
        unsafe {
            (self.api.bitmap_free)(self.ptr.as_ptr());
        }
    }
}
pub(super) struct Chunks<'a> {
    vision: &'a Vision,
    ptr: NonNull<c_void>,
    _bitmaps: Vec<Bitmap<'a>>,
}
impl Drop for Chunks<'_> {
    fn drop(&mut self) {
        // SAFETY: uniquely owned chunks, freed before their bitmaps.
        unsafe {
            (self.vision.api.chunks_free)(self.ptr.as_ptr());
        }
    }
}
impl Chunks<'_> {
    pub(super) fn positions(&self) -> usize {
        // SAFETY: chunks returned by successful tokenization.
        unsafe { (self.vision.api.positions)(self.ptr.as_ptr()).max(0) as usize }
    }
    pub(super) fn evaluate(&self, context: *mut c_void) -> Result<usize, Error> {
        let mut end = 0;
        // SAFETY: caller owns context mutex, projector and chunks are live. Only prefix logits are omitted.
        let rc = unsafe {
            (self.vision.api.eval)(
                self.vision.ptr.as_ptr(),
                context,
                self.ptr.as_ptr(),
                0,
                0,
                super::N_BATCH as i32,
                false,
                &mut end,
            )
        };
        if rc != 0 {
            return Err(model_error(format!(
                "image prefix evaluation failed ({rc})"
            )));
        }
        if end < 0 || end as usize != self.positions() {
            return Err(model_error("image prefix position mismatch"));
        }
        Ok(end as usize)
    }
}
