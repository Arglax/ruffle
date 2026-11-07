package flash.utils {
    import __ruffle__.stub_constructor;

    [Ruffle(InstanceAllocator)]
    public dynamic class Dictionary {
        prototype.toJSON = function(r:String):* {
            return "Dictionary";
        };
        prototype.setPropertyIsEnumerable("toJSON", false);

        public function Dictionary(weakKeys:Boolean = false) {
            if (weakKeys) {
                this.mvsWeakKeys();
            }
        }

        // MVS: weak keys (see dynamic_map.rs and Library::mvs_keep).
        private native function mvsWeakKeys():void;
    }
}
