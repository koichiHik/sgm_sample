
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include "opencv2/opencv.hpp"

#include "sgbm.h"


void recv_file_path(std::string &path) {

  do {
    std::cin >> path;
    std::ifstream ifs(path); 

    if (ifs.is_open()) {
      break;
    }

    std::cout << "The specified file can not be opened. Please enter again. " << std::endl;

  } while (true);
  
}

int recv_int() {

  int val = 0;
  std::string val_str;
  do {

    try {
      std::cin >> val_str;
      val = std::stoi(val_str);
      break;
    } catch (std::exception e) {
      std::cout << "Input data is not integer. Please enter integer." << std::endl;
    }

  } while(true);

  return val;
}

void recv_console_input(std::string &left_path, std::string &right_path, int &disp_range, int &p1, int &p2) {

  std::cout << "Please enter left image path." << std::endl;
  recv_file_path(left_path);

  std::cout << "Please enter right image path." << std::endl;
  recv_file_path(right_path);

  std::cout << "Please specify disparity range." << std::endl;
  disp_range = recv_int();

  std::cout << "Please specify p1." << std::endl;
  p1 = recv_int();

  std::cout << "Please specify p2." << std::endl;
  p2 = recv_int();

  return;
}

int main(int argc, char** argv) {

  std::cout << "SGBM Test Started!" << std::endl;

  std::string left_path, right_path;
  int disp_r, p1, p2;

  std::cout << "0. Load parameters from user." << std::endl;
  recv_console_input(left_path, right_path, disp_r, p1, p2);

  std::cout << "1. Open and load images" << std::endl;
  cv::Mat left = cv::imread(left_path, cv::IMREAD_GRAYSCALE);
  cv::Mat right = cv::imread(right_path, cv::IMREAD_GRAYSCALE);

  std::cout << "2. Initialize class" << std::endl;
  Sgbm sgbm(left.rows, left.cols, disp_r, 3, 20, true, true);
  cv::Mat disp;
  sgbm.compute_disp(left, right, disp);

  return 0;
}
