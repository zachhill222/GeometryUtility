#pragma once

#include "utility/utility.hpp"
#include "geometry/geometry.hpp"
#include "shapes/sphere.hpp"

#include <vector>
#include <fstream>
#include <sstream>

namespace gutil {

	template<int DIM, IsReal T> requires (DIM>0)
	[[nodiscard]] inline std::vector<Sphere<DIM,T>> read_spheres_from_file(const std::string& filename) {
		//expected format: r x y z
		std::vector<Sphere<DIM,T>> result;

		//open file
		std::ifstream file(filename);
		std::string line;

		if( !file.is_open() ) {
			Logger::error("Could not open ", filename);
			return {};
		}

		//read file
		T radius;
		Point<DIM,T> center;

		while (getline(file, line)){
			if (!line.empty() && line[0] != '#'){
				std::istringstream iss(line);

				iss >> radius;
				for (int i=0; i<DIM; ++i) { iss >> center[i]; }

				result.emplace_back(center, radius);
			}
		}

		return result;
	}


	//save a list of spheres to a file
	template<int DIM, IsReal T> requires (DIM>0)
	void write_spheres_to_file(const std::string filename, std::span<const Sphere<DIM,T>> list)	{
		std::ofstream file(filename);

		if ( !file.is_open() ){
			Logger::error("Couldn't write to ", filename);
			file.close();
			return;
		}
		
		std::stringstream buffer;
		for (const auto& sphere : list) {
			buffer << sphere.radius << " " << sphere.center << "\n";
		}
		file << buffer.rdbuf();
		file.close();
	}


}
