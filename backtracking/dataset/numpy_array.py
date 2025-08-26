import pyarrow
from numpy import dtype, float32
from numpy.typing import DTypeLike, NDArray


class NumpyArray(pyarrow.ExtensionScalar):
    @staticmethod
    def from_numpy(array: NDArray):
        spec_type = NumpyArrayType(array.dtype)
        return pyarrow.scalar(
            {"shape": array.shape, "data": array.flatten()}, spec_type.storage_type
        )

    def as_py(self) -> NDArray:
        return (
            self.value["data"]
            .values.to_numpy(zero_copy_only=True)
            .reshape(self.value["shape"].values.to_numpy(zero_copy_only=True))
        )


class NumpyArrayType(pyarrow.ExtensionType):
    _dtype: DTypeLike

    EXTENSION_TYPE_NAME = "numpy_array"

    @property
    def dtype(self):
        return self._dtype

    def ensure_registered(self):
        # Always unregister if exists, to allow reloading it if the code changes
        try:
            pyarrow.unregister_extension_type(self.EXTENSION_TYPE_NAME)
        except Exception:
            pass
        pyarrow.register_extension_type(self)

    def __init__(self, _dtype: DTypeLike = float32):
        self._dtype = _dtype
        super().__init__(
            pyarrow.struct(
                {
                    "shape": pyarrow.list_(pyarrow.int64()),
                    "data": pyarrow.list_(pyarrow.from_numpy_dtype(_dtype)),
                }
            ),
            self.EXTENSION_TYPE_NAME,
        )

    def __arrow_ext_serialize__(self) -> bytes:
        return str(dtype(self._dtype)).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return cls(dtype(serialized.decode()))

    def __arrow_ext_scalar_class__(self):
        return NumpyArray
