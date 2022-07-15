




using Tensorflow.NumPy;

namespace PaddleOCR;

public static class NDArrayExtensions {
    public static NDArray Copy(this NDArray a) {
        var toReturn = new NDArray(a.ToByteArray(), a.shape, a.dtype);
        return toReturn;
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