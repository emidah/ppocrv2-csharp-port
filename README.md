# C# port of PPOCRv2

This is a direct C#/.Net port of https://github.com/PaddlePaddle/Paddle2ONNX/tree/develop/model_zoo/ocr, using the ch+en models provided.

This application utilizes SharpCV, OpenCVSharp and the Microsoft ONNX runtime to recreate the same functionality.

Everything necessary for running the code is included in the repository. DemoApp contains a simple command line application that can be used to call the library like ```PPOCRv2DemoApp.exe --file filename.png``` 

## TODO:

- Fix bugs due to float/int conversion

- Improve code quality (the direct port is is very... pythonic)

- Do something about pyclipper

- Remove SharpCV dependency
