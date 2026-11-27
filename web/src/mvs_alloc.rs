//! MVS: counts the bytes Rust has handed out and not freed, so the page can tell live data from
//! memory the allocator keeps after a free. WebAssembly memory never shrinks: the difference between
//! the module's size and the live bytes is what the game once needed and no longer uses.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

pub struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

/// Live bytes by allocation size: under 256 B, 4 KB, 64 KB, 1 MB, and 1 MB or more. Small
/// allocations are game objects and strings; the large ones are pixel buffers, SWF data and caches.
static BUCKETS: [AtomicUsize; 5] = [const { AtomicUsize::new(0) }; 5];
static COUNTS: [AtomicUsize; 5] = [const { AtomicUsize::new(0) }; 5];

#[inline]
fn bucket(size: usize) -> usize {
    match size {
        0..256 => 0,
        256..4096 => 1,
        4096..65536 => 2,
        65536..1_048_576 => 3,
        _ => 4,
    }
}

#[inline]
fn add(size: usize) {
    let b = bucket(size);
    BUCKETS[b].fetch_add(size, Relaxed);
    COUNTS[b].fetch_add(1, Relaxed);
}

#[inline]
fn remove(size: usize) {
    let b = bucket(size);
    BUCKETS[b].fetch_sub(size, Relaxed);
    COUNTS[b].fetch_sub(1, Relaxed);
}

#[inline]
fn grow(by: usize) {
    let now = LIVE.fetch_add(by, Relaxed) + by;
    PEAK.fetch_max(now, Relaxed);
}

/// Allocations of this size or more carry a small header naming the player phase that made them
/// (ruffle_render::mvs_phase), so the report can say which part of the player holds the big buffers.
const BIG: usize = 65536;

static PHASE_BYTES: [AtomicUsize; ruffle_render::mvs_phase::COUNT] =
    [const { AtomicUsize::new(0) }; ruffle_render::mvs_phase::COUNT];

#[inline]
fn header(layout: Layout) -> usize {
    layout.align().max(16)
}

unsafe fn alloc_big(layout: Layout, zeroed: bool) -> *mut u8 {
    let h = header(layout);
    let Ok(outer) = Layout::from_size_align(layout.size() + h, h) else {
        return std::ptr::null_mut();
    };
    let base = unsafe {
        if zeroed {
            System.alloc_zeroed(outer)
        } else {
            System.alloc(outer)
        }
    };
    if base.is_null() {
        return base;
    }
    let phase = ruffle_render::mvs_phase::current().min(ruffle_render::mvs_phase::COUNT as u8 - 1);
    unsafe { base.write(phase) };
    PHASE_BYTES[phase as usize].fetch_add(layout.size(), Relaxed);
    unsafe { base.add(h) }
}

unsafe fn dealloc_big(ptr: *mut u8, layout: Layout) {
    let h = header(layout);
    let base = unsafe { ptr.sub(h) };
    let phase = unsafe { base.read() } as usize;
    if let Some(bytes) = PHASE_BYTES.get(phase) {
        bytes.fetch_sub(layout.size(), Relaxed);
    }
    let outer = unsafe { Layout::from_size_align_unchecked(layout.size() + h, h) };
    unsafe { System.dealloc(base, outer) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = if layout.size() >= BIG {
            unsafe { alloc_big(layout, false) }
        } else {
            unsafe { System.alloc(layout) }
        };
        if !ptr.is_null() {
            grow(layout.size());
            add(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = if layout.size() >= BIG {
            unsafe { alloc_big(layout, true) }
        } else {
            unsafe { System.alloc_zeroed(layout) }
        };
        if !ptr.is_null() {
            grow(layout.size());
            add(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if layout.size() >= BIG {
            unsafe { dealloc_big(ptr, layout) };
        } else {
            unsafe { System.dealloc(ptr, layout) };
        }
        LIVE.fetch_sub(layout.size(), Relaxed);
        remove(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if layout.size() < BIG && new_size < BIG {
            let new = unsafe { System.realloc(ptr, layout, new_size) };
            if !new.is_null() {
                remove(layout.size());
                add(new_size);
                if new_size >= layout.size() {
                    grow(new_size - layout.size());
                } else {
                    LIVE.fetch_sub(layout.size() - new_size, Relaxed);
                }
            }
            return new;
        }
        // Either side is big (has a header): move it.
        let Ok(new_layout) = Layout::from_size_align(new_size, layout.align()) else {
            return std::ptr::null_mut();
        };
        let new = unsafe { self.alloc(new_layout) };
        if !new.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(ptr, new, layout.size().min(new_size));
                self.dealloc(ptr, layout);
            }
        }
        new
    }
}

/// "render 12 MB, script 40 MB, ..." for big (64 KB+) allocations still alive.
pub fn phases() -> String {
    let mut out = String::new();
    for (i, name) in ruffle_render::mvs_phase::NAMES.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        let mb = PHASE_BYTES[i].load(Relaxed) as f64 / 1_048_576.0;
        out.push_str(&format!("{name} {mb:.0} MB"));
    }
    out
}

/// Bytes allocated and not yet freed.
pub fn live() -> usize {
    LIVE.load(Relaxed)
}

/// Highest `live()` seen since the module started.
pub fn peak() -> usize {
    PEAK.load(Relaxed)
}

/// "small 120 MB (1.2M), ..." for the memory report.
pub fn buckets() -> String {
    const NAMES: [&str; 5] = ["<256B", "<4KB", "<64KB", "<1MB", ">=1MB"];
    let mut out = String::new();
    for (i, name) in NAMES.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        let mb = BUCKETS[i].load(Relaxed) as f64 / 1_048_576.0;
        let n = COUNTS[i].load(Relaxed);
        out.push_str(&format!("{name} {mb:.0} MB/{n}"));
    }
    out
}
