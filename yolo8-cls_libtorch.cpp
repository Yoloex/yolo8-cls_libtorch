// yolo8-cls_libtorch.cpp : Defines the entry point for the application.
//

#include "yolo8-cls_libtorch.h"

using namespace std;
using namespace torch;

typedef unsigned int uint;
const TensorOptions options = TensorOptions().dtype(kByte);

at::Tensor ToTensor(uint8_t*img, int width, int height, int channels)
{
	at::Tensor tensor_image;

	try {
		tensor_image = torch::from_blob(img, { height, width, channels }, options);
	}
	catch (const c10::Error e) {
		cerr << e.what() << endl;
	}

	return tensor_image;
}

int main(int argc, char* argv[])
{
	string model_path = argv[1], img_path = argv[2];
	jit::script::Module module;

	try {
		module = jit::load(model_path);
	}
	catch (const c10::Error& e) {
		cerr << e.what() << endl;
		cerr << "Error loading the model" << endl;
		return -1;
	}

	std::vector<uint8_t> out;
	uint width;
	uint height;
	uint channels_in_file;
	uint desired_channels = 3;

	int ret = fpng::fpng_decode_file(img_path.c_str(), out, width, height, channels_in_file, desired_channels);

	if (!ret) {
		cerr << "Error reading the image" << endl;
		return -2;
	}

	uint8_t* buffer = (uint8_t*)malloc(out.size());
	memcpy(buffer, out.data(), out.size());
	at::Tensor input = ToTensor(buffer, width, height, desired_channels);
	input = input.permute({ 2, 0, 1 }).unsqueeze(0);

	Tensor input_tensor;

	try {
		input_tensor = nn::functional::interpolate(
			input,
			nn::functional::InterpolateFuncOptions()
			.mode(kBilinear)
			.align_corners(false)
			.size(vector<int64_t>({ 224, 224 }))
		);
	}
	catch (const c10::Error e) {
		cerr << e.what() << endl;
		return -3;
	}

	input_tensor = input_tensor.toType(kFloat32).div(255);

	Tensor res;

	try{
			res = module.forward({ input_tensor }).toTensor();
	}
	catch (const c10::Error e) {
		cerr << e.what() << endl;
	}
	
	vector<float> result(res.data_ptr<float>(), res.data_ptr<float>() + res.numel());

	cout << result << endl;

	return 0;
}
