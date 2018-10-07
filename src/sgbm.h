
#include "opencv2/opencv.hpp"

typedef std::vector<std::vector<std::vector<unsigned long> > > cost_3d_array;
typedef std::vector<std::vector<std::vector<std::vector<unsigned long> > > > cost_4d_array;

class ScanLine {
public:
  ScanLine(int drow, int dcol, bool posdir) {
    this->drow = drow;
    this->dcol = dcol;
    this->posdir = posdir;
  }
  bool posdir;
  int drow, dcol;
};

class ScanLines8 {
public:
  ScanLines8() {
    this->path8.push_back(ScanLine(1, 1, true));
    this->path8.push_back(ScanLine(1, 0, true));
    this->path8.push_back(ScanLine(1, -1, true));
    this->path8.push_back(ScanLine(0, -1, false));
    this->path8.push_back(ScanLine(-1, -1, false));
    this->path8.push_back(ScanLine(-1, 0, false));
    this->path8.push_back(ScanLine(-1, 1, false));
    this->path8.push_back(ScanLine(0, 1, true));
  }
  
  std::vector<ScanLine> path8;
};

class Sgbm {

public:
  Sgbm(int rows, int cols, int d_range, unsigned short p1, unsigned short p2);

  ~Sgbm();

  void reset_buffer();

  void compute_disp(cv::Mat &left, cv::Mat &right, cv::Mat &disp);

  void calc_matching_cost();

  unsigned short aggregate_cost(int row, int col, int depth, int path, cost_3d_array &pix_cost, cost_4d_array &agg_cost);

  void aggregate_cost_for_each_scanline(cost_3d_array &pix_cost, cost_4d_array &agg_cost, cost_3d_array &sum_cost);

  void calc_disparity(cost_3d_array &sum_cost, cv::Mat &disp_img);

  void census_transform(cv::Mat &img, cv::Mat &census);

  void calc_pixel_cost(cv::Mat &census_l, cv::Mat &census_r, cost_3d_array &pix_cost);

  unsigned char calc_hamming_dist(unsigned char val_l, unsigned char val_r);

public:

  int rows, cols, d_range, scanpath;
  unsigned short p1, p2;
  cv::Mat *census_l, *census_r, *disp_img;
  cost_3d_array pix_cost;
  cost_4d_array agg_cost;
  cost_3d_array sum_cost;
  ScanLines8 scanlines;
};
