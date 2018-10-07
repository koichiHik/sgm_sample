
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

  cv::GaussianBlur(left, left, cv::Size(3, 3), 3);
  cv::GaussianBlur(right, right, cv::Size(3, 3), 3);
  cv::imshow("Left Original", left);
  cv::imshow("Right Original", right);
  cv::waitKey(0);

  std::cout << "2. Initialize class" << std::endl;
  Sgbm sgbm(left.rows, left.cols, disp_r, 3, 20);

  std::cout << "3. Census transform" << std::endl;
  sgbm.census_transform(left, *sgbm.census_l);
  sgbm.census_transform(right, *sgbm.census_r);

  cv::imshow("Left Census Trans", *sgbm.census_l);
  cv::imshow("Right Census Trans", *sgbm.census_r);
  cv::waitKey(0);

  std::cout << "4. Matching Cost Calculation" << std::endl;
  sgbm.calc_pixel_cost(*sgbm.census_l, *sgbm.census_r, sgbm.pix_cost);

  std::cout << "5. Cost Aggregation" << std::endl;
  sgbm.aggregate_cost_for_each_scanline(sgbm.pix_cost, sgbm.agg_cost, sgbm.sum_cost);

  std::cout << "6. Disparity Image" << std::endl;
  sgbm.calc_disparity(sgbm.sum_cost, *sgbm.disp_img);

  std::cout << "7. Visualize Disparity Image" << std::endl;
  cv::Mat disp;
  sgbm.disp_img->convertTo(disp, CV_8U, 256.0/disp_r);
  cv::imshow("Sgbm Result", disp);
  cv::waitKey(0);

  return 0;
}
