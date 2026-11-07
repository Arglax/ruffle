//! Special handling for AVM2 orphan objects

use crate::context::UpdateContext;
use crate::display_object::{DisplayObject, DisplayObjectWeak, TDisplayObject};
use gc_arena::{Collect, Mutation};
use std::rc::Rc;

/// The list of 'orphan' objects - these objects have no parent,
/// so we need to manually run their frames in `run_all_phases_avm2` to match
/// Flash's behavior. Clips are added to this list with `add_orphan_movie`.
/// and are removed automatically by `cleanup_dead_orphans`.
///
/// We store `DisplayObjectWeak`, since we don't want to keep these objects
/// alive if they would otherwise be garbage-collected. The movie will
/// stop ticking whenever garbage collection runs if there are no more
/// strong references around (this matches Flash's behavior).
#[derive(Collect)]
#[collect(no_drop)]
pub struct OrphanManager<'gc> {
    orphans: Rc<Vec<DisplayObjectWeak<'gc>>>,
    /// MVS: per orphan, whether it was taken off the stage (not one that never was on it).
    #[collect(require_static)]
    removed: Rc<Vec<bool>>,
}

impl<'gc> OrphanManager<'gc> {
    fn orphans_mut(&mut self) -> &mut Vec<DisplayObjectWeak<'gc>> {
        Rc::make_mut(&mut self.orphans)
    }

    /// Adds a `MovieClip` to the orphan list. In AVM2, movies advance their
    /// frames even when they are not on a display list. Unfortunately,
    /// multiple SWFS rely on this behavior, so we need to match Flash's
    /// behavior. This should not be called manually - `movie_clip` will
    /// call it when necessary.
    pub fn add_orphan_obj(&mut self, dobj: DisplayObject<'gc>) {
        self.add_orphan(dobj, false);
    }

    /// MVS: an orphan that was taken off the stage (see `mvs_interp::freeze_orphans`).
    pub fn add_removed_orphan_obj(&mut self, dobj: DisplayObject<'gc>) {
        self.add_orphan(dobj, true);
    }

    fn add_orphan(&mut self, dobj: DisplayObject<'gc>, removed: bool) {
        // Note: comparing pointers is correct because GcWeak keeps its allocation alive,
        // so the pointers can't overlap by accident.
        if let Some(i) = self
            .orphans
            .iter()
            .position(|d| std::ptr::eq(d.as_ptr(), dobj.as_ptr()))
        {
            if removed && !self.removed[i] {
                Rc::make_mut(&mut self.removed)[i] = true;
            }
        } else {
            self.orphans_mut().push(dobj.downgrade());
            Rc::make_mut(&mut self.removed).push(removed);
        }
    }

    /// MVS: orphans kept, and how many of them were taken off the stage.
    pub fn mvs_counts(&self) -> (usize, usize) {
        (self.orphans.len(), self.removed.iter().filter(|r| **r).count())
    }

    pub fn each_orphan_obj(
        context: &mut UpdateContext<'gc>,
        mut f: impl FnMut(DisplayObject<'gc>, &mut UpdateContext<'gc>),
    ) {
        // Clone the Rc before iterating over it. Any modifications must go through
        // `Rc::make_mut` in `orphan_objects_mut`, which will leave this `Rc` unmodified.
        // This ensures that any orphan additions/removals done by `f` will not affect
        // the iteration in this method.
        let orphan_objs: Rc<_> = context.orphan_manager.orphans.clone();
        let removed: Rc<_> = context.orphan_manager.removed.clone();
        // MVS: clips the game took off the stage are invisible; with the switch on they are left
        // as they are instead of being animated every frame until memory is next cleaned up.
        let skip_removed = crate::display_object::mvs_interp::freeze_orphans();

        for (i, orphan) in orphan_objs.iter().enumerate() {
            if skip_removed && removed.get(i).copied().unwrap_or(false) {
                continue;
            }
            if let Some(dobj) = valid_orphan(*orphan, context.gc()) {
                f(dobj, context);
            }
        }
    }

    /// Called at the end of `run_all_phases_avm2` - removes any movies
    /// that have been garbage collected, or are no longer orphans
    /// (they've since acquired a parent).
    pub fn cleanup_dead_orphans(&mut self, mc: &Mutation<'gc>) {
        let mut keep = Vec::with_capacity(self.orphans.len());
        let removed = self.removed.clone();
        let mut i = 0;
        self.orphans_mut().retain(|d| {
            let was_removed = removed.get(i).copied().unwrap_or(false);
            i += 1;
            let alive = {
            if let Some(dobj) = valid_orphan(*d, mc) {
                // All clips that become orphaned (have their parent removed, or start out with no parent)
                // get added to the orphan list. However, there's a distinction between clips
                // that are removed from a RemoveObject tag, and clips that are removed from ActionScript.
                //
                // Clips removed from a RemoveObject tag only stay on the orphan list until the end
                // of the frame - this lets them run a framescript (with 'this.parent == null')
                // before they're removed. After that, they're removed from the orphan list,
                // and will not be run in any way.
                //
                // Clips removed from ActionScript stay on the orphan list, and will be run
                // indefinitely (if there are no remaining strong references, they will eventually
                // be garbage collected).
                //
                // To detect this, we check 'placed_by_avm2_script'. This flag get set to 'true'
                // for objects constructed from ActionScript, and for objects moved around
                // in the timeline (add/remove child, swap depths) by ActionScript. A
                // RemoveObject tag will only affect objects instantiated by the timeline,
                // which have not been moved in the displaylist by ActionScript. Therefore,
                // any orphan we see that has 'placed_by_avm2_script()' should stay on the orphan
                // list, because it was not removed by a RemoveObject tag.
                dobj.placed_by_avm2_script()
            } else {
                false
            }
            };
            if alive {
                keep.push(was_removed);
            }
            alive
        });
        self.removed = Rc::new(keep);
    }
}

impl<'gc> Default for OrphanManager<'gc> {
    fn default() -> Self {
        Self {
            orphans: Rc::new(Vec::new()),
            removed: Rc::new(Vec::new()),
        }
    }
}

/// If the provided `DisplayObjectWeak` should have frames run, returns
/// Some(clip) with an upgraded `MovieClip`.
/// If this returns `None`, the entry should be removed from the orphan list.
fn valid_orphan<'gc>(
    dobj: DisplayObjectWeak<'gc>,
    mc: &Mutation<'gc>,
) -> Option<DisplayObject<'gc>> {
    dobj.upgrade(mc).filter(|dobj| dobj.parent().is_none())
}

/// MVS: up to three orphans taken off the stage (AQW avatars first), as (display object, AS3
/// object) pointers with their class, for `Player::mvs_find_holders`.
pub fn mvs_holder_targets<'gc>(context: &mut UpdateContext<'gc>) -> Vec<(*const (), *const (), String)> {
    use crate::avm2::object::TObject;
    let orphans = context.orphan_manager.orphans.clone();
    let removed = context.orphan_manager.removed.clone();
    let mut found: Vec<(*const (), *const (), String)> = Vec::new();
    for pass in 0..2 {
        for (i, weak) in orphans.iter().enumerate() {
            if found.len() >= 3 || !removed.get(i).copied().unwrap_or(false) {
                continue;
            }
            let Some(dobj) = valid_orphan(*weak, context.gc()) else {
                continue;
            };
            let class = dobj
                .object2()
                .map(|o| o.instance_class().name().local_name().to_string())
                .unwrap_or_else(|| "-".to_string());
            if (pass == 0) != (class == "AvatarMC") {
                continue;
            }
            let obj = dobj.object2().map(|o| o.as_ptr() as *const ()).unwrap_or(std::ptr::null());
            found.push((dobj.as_ptr() as *const (), obj, class));
        }
    }
    found
}

/// MVS: a census of what the frame loop visits each frame: display objects on the stage (shown
/// and hidden) and orphans (removed from the stage but still run, like Flash), with the classes
/// that hold the most of them. For finding where frame time goes when little is drawn.
pub fn mvs_census<'gc>(context: &mut UpdateContext<'gc>) -> String {
    use crate::avm2::object::TObject;
    use crate::display_object::TDisplayObjectContainer;
    use std::collections::HashMap;

    fn class_of(d: DisplayObject<'_>) -> String {
        d.object2()
            .map(|o| o.instance_class().name().local_name().to_string())
            .unwrap_or_else(|| "-".to_string())
    }

    // Returns the subtree size; records (instances, objects under them) per named class.
    fn walk<'gc>(
        d: DisplayObject<'gc>,
        hidden: bool,
        shown_hidden: &mut [u32; 2],
        classes: &mut HashMap<String, (u32, u32)>,
    ) -> u32 {
        let hidden = hidden || !d.visible();
        shown_hidden[hidden as usize] += 1;
        let mut n = 1;
        if let Some(c) = d.as_container() {
            for child in c.iter_render_list() {
                n += walk(child, hidden, shown_hidden, classes);
            }
        }
        let name = class_of(d);
        if !matches!(name.as_str(), "MovieClip" | "Sprite" | "Shape" | "-" | "TextField" | "SimpleButton") {
            let e = classes.entry(name).or_insert((0, 0));
            e.0 += 1;
            e.1 += n;
        }
        n
    }

    fn top(classes: HashMap<String, (u32, u32)>) -> String {
        let mut v: Vec<_> = classes.into_iter().collect();
        v.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
        v.iter()
            .take(12)
            .map(|(k, (n, sub))| format!("{k} x{n} ({sub})"))
            .collect::<Vec<_>>()
            .join(", ")
    }

    let mut stage_counts = [0u32; 2];
    let mut stage_classes = HashMap::new();
    walk(context.stage.into(), false, &mut stage_counts, &mut stage_classes);

    let mut orphan_counts = [0u32; 2];
    let mut orphan_classes = HashMap::new();
    let mut orphans = 0;
    let list = context.orphan_manager.orphans.clone();
    for o in list.iter() {
        if let Some(d) = valid_orphan(*o, context.gc()) {
            orphans += 1;
            walk(d, false, &mut orphan_counts, &mut orphan_classes);
        }
    }
    format!(
        "stage: {} shown, {} hidden | orphans: {} ({} objects in them, list {})\nstage classes: {}\norphan classes: {}",
        stage_counts[0],
        stage_counts[1],
        orphans,
        orphan_counts[0] + orphan_counts[1],
        list.len(),
        top(stage_classes),
        top(orphan_classes)
    )
}
