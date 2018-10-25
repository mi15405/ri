#include "fann.h"

// dodati fleg '-lfann' pri kompilaciji (za linkovanje) 

int main()
{
	const unsigned int nbLayers = 2;
	const unsigned int nbInput = 2;
	const unsigned int nbOutput = 1;
	const unsigned int maxIterations = 10000;
	const float maxError = (const float) 0.0001;
	const unsigned int reportAfter = 5;

	// broj slojeva, broj ulaza, broj izlaza 
	struct fann* ann = fann_create_standard(nbLayers, nbInput, nbOutput);

	// Aktivaciona funkcija
	fann_set_activation_function_output(ann, FANN_SIGMOID);

	// Fajl, maks broj iteracija, nakon koliko iteracija daje izvestaj, maksimalna greska
	fann_train_on_file(ann, "and_input.txt", maxIterations, reportAfter, maxError);

	// Cuvamo neuronsku mrezu
	fann_save(ann, "and_net.txt");
	
	// Brisemo neuronsku mrezu
	fann_destroy(ann);
	
	return 0;
}
