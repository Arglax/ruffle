//! MVS: heap inspection for finding what keeps an object alive.

use crate::{Collect, Gc, GcWeak, collect::Trace};
use alloc::vec::Vec;

/// Collects the strong GC pointers a value holds (weak ones are skipped).
#[derive(Default)]
pub struct EdgeTracer {
    pub strong: Vec<*const ()>,
}

impl<'gc> Trace<'gc> for EdgeTracer {
    fn trace_gc(&mut self, gc: Gc<'gc, ()>) {
        self.strong.push(Gc::as_ptr(gc));
    }

    fn trace_gc_weak(&mut self, _gc: GcWeak<'gc, ()>) {}
}

/// The strong GC pointers held directly by `root` (any `Collect` value, such as an arena root).
pub fn root_edges<'gc, R: Collect<'gc> + ?Sized>(root: &R) -> Vec<*const ()> {
    let mut t = EdgeTracer::default();
    root.trace(&mut t);
    t.strong
}

/// The strong GC pointers held by the GC value at `ptr`.
///
/// # Safety
/// `ptr` must point at a live GC value reached through strong pointers from the root.
pub unsafe fn edges(ptr: *const ()) -> Vec<*const ()> {
    let mut t = EdgeTracer::default();
    unsafe { crate::types::mvs_edges(ptr, &mut t) };
    t.strong
}

/// The type name of the GC value at `ptr`.
///
/// # Safety
/// As for `edges`.
pub unsafe fn type_name(ptr: *const ()) -> &'static str {
    unsafe { crate::types::mvs_type_name(ptr) }
}
