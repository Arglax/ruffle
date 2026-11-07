pub use crate::avm2::object::dictionary_allocator;

/// MVS: `new Dictionary(true)`: object keys are held weakly, as in Flash.
pub fn mvs_weak_keys<'gc>(
    activation: &mut crate::avm2::Activation<'_, 'gc>,
    this: crate::avm2::Value<'gc>,
    _args: crate::avm2::function::FunctionArgs<'_, 'gc>,
) -> Result<crate::avm2::Value<'gc>, crate::avm2::Error<'gc>> {
    use crate::avm2::object::TObject as _;
    if let Some(this) = this.as_object() {
        this.base().values().set_weak_keys();
        activation.context.library.mvs_register_weak_dict(this);
    }
    Ok(crate::avm2::Value::Undefined)
}
