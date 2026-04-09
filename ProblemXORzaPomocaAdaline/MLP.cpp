#include <iostream>
#include <Eigen/Dense>
#include <random>

class Layer{
public:
	Eigen::MatrixXd weights; // macierz wag
	Eigen::VectorXd biases; // wwektor biasów jeden na każdy neuron
	Eigen::VectorXd last_input; // Pamięć do ostatniego wejścia do backpropagation
	Eigen::VectorXd last_output; // Pamięć dla wyniku po sigmoidzie

	Layer(int in_size, int out_size, int seed) {
		std::mt19937 gen(seed);
		std::normal_distribution<double> dist(0.0, 0.5);


		weights.resize(in_size, out_size);
		biases = Eigen::VectorXd::Zero(out_size);

		for (int i = 0; i < weights.rows(); ++i) {
			for (int j = 0; j < weights.cols(); ++j) {
				weights(i, j) = dist(gen);
			}
		}
	}


	double sigmoid(double z) {
		return 1.0 / (1.0 + std::exp(-z));
	}

	// Propagacja w przód
	// Decyduje czy trafi do nastepnej warstwy czy jest ostatecznym wynikiem sieci
	Eigen::VectorXd forward(const Eigen::VectorXd& input) {
		last_input = input; // zapamiętanie wejścia aby przy backpropagition wiedzieć gdzie był błąd

		Eigen::VectorXd z = (input.transpose() * weights).transpose() + biases;

		last_output = z.unaryExpr([this](double v) {return sigmoid(v); });

		return last_output;
	}
};

class MLP {
	std::vector<Layer> layers;
public:
	void add_layer(int in_size, int out_size) {
		layers.emplace_back(in_size, out_size, 123 + layers.size());
	}
	Eigen::VectorXd predict(Eigen::VectorXd x) {
		for (auto& layer : layers) {
			x = layer.forward(x);
		}
		return x;
	}


	// Backpropagition uczenie maszynowe
	void train(const Eigen::VectorXd& input, const Eigen::VectorXd& target, double eta) {
		Eigen::VectorXd output = predict(input);
		Eigen::VectorXd delta = (target - output).array() * output.array() * (1.0 - output.array());

		//pętla od tyłu
		for (int i = layers.size() - 1; i >= 0; --i) {
			Layer& L = layers[i];
			Eigen::MatrixXd weight_update = eta * (L.last_input * delta.transpose());
			L.weights += weight_update;
			L.biases += eta * delta;
			if (i > 0) {
				Eigen::VectorXd prev_output = layers[i - 1].last_output;
				delta = (L.weights * delta).array() * prev_output.array() * (1.0 - prev_output.array());
			}
		}
	}
};

int main() {
	MLP net;
	net.add_layer(2, 2);
	net.add_layer(2, 1);

	// Dane do anuki XOR
	std::vector<Eigen::VectorXd> inputs = {
		(Eigen::VectorXd(2) << 0, 0).finished(),
		(Eigen::VectorXd(2) << 0, 1).finished(),
		(Eigen::VectorXd(2) << 1, 0).finished(),
		(Eigen::VectorXd(2) << 1, 1).finished()
	};
	std::vector<Eigen::VectorXd> targets = {
		(Eigen::VectorXd(1) << 0).finished(),
		(Eigen::VectorXd(1) << 1).finished(),
		(Eigen::VectorXd(1) << 1).finished(),
		(Eigen::VectorXd(1) << 0).finished()
	};

	// Pętla uczenia
	for (int epoch = 0; epoch < 10000; ++epoch) {
		for (size_t i = 0; i < inputs.size(); ++i) {
			net.train(inputs[i], targets[i], 0.1); // eta = 0.1
		}
	}

	std::cout << "Wyniki po nauce (XOR):" << std::endl;
	for (auto& in : inputs) {
		std::cout << in.transpose() << " => " << net.predict(in) << std::endl;
	}

	return 0;
}
