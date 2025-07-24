/*
 * main.cpp
 *
 *  Created on: May 20, 2025
 *      Author: voodoocode
 */



#include "main.h"

//path = "/mnt/data/Professional/LMU/code/Beta_prevalence/spatial/"
//file_name = "vol.vti"

int main(int argc, char* argv[]) {

	std::vector<std::string> args;
	std::copy(argv + 1, argv + argc, std::back_inserter(args));

	std::string path;
	std::string f_name;

	if (args[0] == "-path") {
		path = args[1];
	} else if (args[0] == "-fname") {
		f_name = args[1];
	} else {
		std::cout << "Error, input argument " << args[0] << " is not valid. Must be either '-path <string>' or '-fname <string>'.";
		return -1;
	}

	if (args[2] == "-path") {
		path = args[3];
	} else if (args[2] == "-fname") {
		f_name = args[3];
	} else {
		std::cout << "Error, input argument " << args[2] << " is not valid. Must be either '-path <string>' or '-fname <string>'.";
		return -1;
	}

	vtkSmartPointer<vtkXMLImageDataReader> reader = vtkSmartPointer<vtkXMLImageDataReader>::New();
	//reader->SetFileName("/mnt/data/Professional/LMU/code/Beta_prevalence/spatial/tmp.vti");
	reader->SetFileName((path + f_name).c_str());
	reader->Update();

	vtkImageData* meta = reader->GetOutput();
	vtkCellData* cellData = meta->GetCellData();

	vtkDataArray* color = cellData->GetScalars("values-I");
	vtkDataArray* opacity = cellData->GetScalars("values-II");

	int* dims = meta->GetDimensions();
	//Blender cannot read these as of Blender 4.4
	//double* spacing = meta->GetSpacing();
	//double* origin = meta->GetOrigin();

	openvdb::initialize();
	openvdb::FloatGrid::Ptr color_grid = openvdb::FloatGrid::create();
	openvdb::FloatGrid::Ptr opacity_grid = openvdb::FloatGrid::create();
	openvdb::FloatGrid::Accessor color_accessor = color_grid->getAccessor();
	openvdb::FloatGrid::Accessor opacity_accessor = opacity_grid->getAccessor();
	color_grid->setName("temperature");
	opacity_grid->setName("flame");

    for (int z = 0; z < dims[2] - 1; ++z) {
    	for (int y = 0; y < dims[1] - 1; ++y) {
    		for (int x = 0; x < dims[0] - 1; ++x) {
    			int loc = x + y * (dims[0] - 1) + z * (dims[0] - 1) * (dims[1] - 1);
    			color_accessor.setValue(openvdb::Coord(x, y, z), color->GetComponent(loc, 0));
    			opacity_accessor.setValue(openvdb::Coord(x, y, z), opacity->GetComponent(loc, 0));
    		}
		}
	}

    std::string raw_f_name = f_name.substr(0, f_name.find_last_of("."));
	openvdb::io::File vdb_file(path + raw_f_name + ".vdb");
	vdb_file.write({color_grid, opacity_grid});
	vdb_file.close();

	std::cout << "Terminated successfully." << std::endl;

	return 0;
}
