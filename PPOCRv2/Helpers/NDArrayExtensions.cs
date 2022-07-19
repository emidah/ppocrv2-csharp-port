using Tensorflow;
using Tensorflow.NumPy;

namespace PPOCRv2.Helpers;

public static class NdArrayExtensions {
    public static NDArray Copy(this NDArray a) {
        var toReturn = new NDArray(a.ToByteArray(), a.shape, a.dtype);
        return toReturn;
    }

    public static NDArray FromArray(NDArray[] arrays) {
        var newShape = arrays[0].shape.as_int_list().Prepend(arrays.Length).ToArray();
        var toReturn = new NDArray(new Shape(newShape), arrays[0].dtype);
        for (var i = 0; i < arrays.Length; i++) {
            toReturn[i] = arrays[i];
        }

        return toReturn;
    }

    public static NDArray NdMin(this NDArray nd) {
        switch (nd.dtype) {
            case TF_DataType.TF_FLOAT: {
                var newArray = nd.ToArray<float>();
                var min = newArray.Min();
                return new NDArray(min);
            }
            case TF_DataType.TF_INT32: {
                var newArray = nd.ToArray<int>();
                var min = newArray.Min();
                return new NDArray(min);
            }
            case TF_DataType.TF_INT8: {
                var newArray = nd.ToArray<sbyte>();
                var min = newArray.Min();
                return new NDArray(min);
            }
            default: throw new NotImplementedException();
        }
    }

    public static NDArray NdMax(this NDArray nd) {
        switch (nd.dtype) {
            case TF_DataType.TF_FLOAT: {
                var newArray = nd.ToArray<float>();
                var min = newArray.Max();
                return new NDArray(min);
            }
            case TF_DataType.TF_INT32: {
                var newArray = nd.ToArray<int>();
                var min = newArray.Max();
                return new NDArray(min);
            }
            case TF_DataType.TF_INT8: {
                var newArray = nd.ToArray<sbyte>();
                var min = newArray.Max();
                return new NDArray(min);
            }
            default: throw new NotImplementedException();
        }
    }

    public static NDArray AsNdArrayOfType(this Tensor t, TF_DataType type) {
        var x = new NDArray(t).astype(type);
        return x;
    }
}