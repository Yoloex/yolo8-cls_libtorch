// yolo8-cls_libtorch.cpp : Defines the entry point for the application.
//

#include "yolo8-cls_libtorch.h"

using namespace std;
using namespace torch;
using namespace lodepng;

typedef unsigned int uint;
typedef unsigned char uchar;

at::Tensor ToTensor(uchar *img, int width, int height, int channels)
{
	at::Tensor tensor_image;

	try {
		tensor_image = torch::from_blob(img, { height, width, channels }, torch::kByte);
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

	vector<uchar> out;
	uint width;
	uint height;
	uint desired_channels = 4;

	uint ret = decode(out, width, height, img_path.c_str());
	
	if (ret) {
		cerr << "Error reading the image" << endl;
		return -2;
	}
	
	uchar* buffer = (uchar*)malloc(out.size());
	memcpy(buffer, out.data(), out.size());

	at::Tensor input = ToTensor(buffer, width, height, desired_channels);
	
	input = input.permute({ 2, 0, 1 }).unsqueeze(0);

	Tensor input_tensor;

	try {
		input = nn::functional::interpolate(
			input,
			nn::functional::InterpolateFuncOptions()
			.mode(kBilinear)
			.size(vector<int64_t>({ 224, 224 }))
			.align_corners(true)
		);
	}
	catch (const c10::Error e) {
		cerr << e.what() << endl;
		return -3;
	}

	input_tensor = input.index({ indexing::Slice(), indexing::Slice(indexing::None, 3), indexing::Slice(), indexing::Slice() });

	Tensor res;

	try{
			res = module.forward({ input_tensor.toType(kFloat32).div(255) }).toTensor();
	}
	catch (const c10::Error e) {
		cerr << e.what() << endl;
	}
	
	vector<float> result(res.data_ptr<float>(), res.data_ptr<float>() + res.numel());

	if (result[0] > result[1]) cout << "off" << endl;
	else cout << "on" << endl;

	return 0;
}
