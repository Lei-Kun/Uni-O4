#include <H5Cpp.h>
#include <vector>
#include <string>
#include <filesystem> // For creating directories
#include <chrono>     // For getting the current time
#include <iomanip>    // For put_time
#include <sstream>    // For stringstream

class HDF5Recorder {
private:
    std::string filename_prefix;
    std::vector<std::vector<float>> actions;
    std::vector<std::vector<float>> states;
    H5::H5File* file = nullptr;
    int episode_count = 0;

public:
    HDF5Recorder() {
        // 在构造函数中创建文件夹
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);

        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");

        filename_prefix = "../dataset/" + ss.str() + "/";

        // 创建文件夹
        std::filesystem::create_directories(filename_prefix);
    }
    
    void record_step(std::vector<float> action, std::vector<float> state) {
        actions.push_back(action);
        states.push_back(state);
    }
    void finish_episode(){
        if (file != nullptr) {
            file->close();
            delete file;
            file = nullptr;
            std::cout << "finish the episode." << std::endl;
        }
    }
    void new_episode() {
        // Close the previous file if it's open
        if (file != nullptr) {
            return;
            // file->close();
            // delete file;
            // file = nullptr;
        }
        std::cout << "record episode: " << episode_count << std::endl;
        // Create a new file for this episode
        std::string filename = filename_prefix + std::to_string(episode_count) + ".hdf5";
        file = new H5::H5File(filename, H5F_ACC_TRUNC);

        hsize_t dims[2] = {actions.size(), actions[0].size()};
        float *action_data = new float[dims[0] * dims[1]];
        for(size_t i=0;i<actions.size();i++){
            std::copy(actions[i].begin(), actions[i].end(), action_data+i*actions[i].size());
        }

        H5::DataSpace dataspace = H5::DataSpace(2, dims);
        H5::DataSet actionsDataset = file->createDataSet("actions", H5::PredType::NATIVE_FLOAT, dataspace);
        actionsDataset.write(action_data, H5::PredType::NATIVE_FLOAT);

        dims[0] = states.size();
        dims[1] = states[0].size();
        float *state_data = new float[dims[0] * dims[1]];

        for(size_t i=0;i<states.size();i++){
            std::copy(states[i].begin(), states[i].end(), state_data+i*states[i].size());
        }
        dataspace = H5::DataSpace(2, dims);
        H5::DataSet statesDataset = file->createDataSet("states", H5::PredType::NATIVE_FLOAT, dataspace);
        statesDataset.write(state_data, H5::PredType::NATIVE_FLOAT);

        // Reset the lists
        actions.clear();
        states.clear();
        if (action_data) delete[] action_data;
        if (state_data) delete[] state_data;
        // Increment the episode count
        episode_count++;
    }

    void close() {
        if (file != nullptr) {
            file->close();
            delete file;
            file = nullptr;
        }
    }
    ~HDF5Recorder() {
        close();
    }
};