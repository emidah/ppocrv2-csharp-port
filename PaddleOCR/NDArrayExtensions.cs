using Tensorflow;
using Tensorflow.NumPy;

namespace PaddleOCR;

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
    //public static unsafe NDArray ToNDArray(this Mat src) {
    //    var nd = new NDArray(NPTypeCode.Byte, (1, src., src.Width, src.Type().Channels), false);
    //    new UnmanagedMemoryBlock<byte>(src.DataPointer, nd.size)
    //        .CopyTo(nd.Unsafe.Address);

    //    return nd;
    //}

    //    public static unsafe NDArray WrapWithNDArray(this Mat src) {
    //        Shape shape = (1, src.Height, src.Width, src.Type().Channels);
    //        var storage = new UnmanagedStorage(
    //            new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(src.DataPointer, shape.Size, () => Donothing(src))),
    //            shape); //we pass donothing as it keeps reference to src preventing its disposal by GC
    //        return new NDArray(storage);
    //    }

    //    [MethodImpl(MethodImplOptions.NoOptimization)]
    //    private static void Donothing(Mat m) {
    //        var a = m;
    //    }
}