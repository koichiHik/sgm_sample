
#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"

#include "sgbm.h"

Sgbm::Sgbm(int rows, int cols, int d_range, unsigned short p1, unsigned short p2) {

  this->rows = rows;
  this->cols = cols;
  this->d_range = d_range;
  this->census_l = new cv::Mat(rows, cols, CV_8UC1);
  this->census_r = new cv::Mat(rows, cols, CV_8UC1);
  this->disp_img = new cv::Mat(rows, cols, CV_8UC1);
  this->p1 = p1;
  this->p2 = p2;
  this->scanpath = 8;

  reset_buffer();
}

Sgbm::~Sgbm() 
{
  delete this->census_l;
  delete this->census_r;
}

void Sgbm::reset_buffer() 
{

  *(this->census_l) = 0;
  *(this->census_r) = 0;

  // Resize vector for Pix Cost
  this->pix_cost.resize(this->rows);
  this->sum_cost.resize(this->rows);
  for (int row = 0; row < this->rows; row++) {
    this->pix_cost[row].resize(this->cols);
    this->sum_cost[row].resize(this->cols);
    for (int col = 0; col < this->cols; col++) {
      this->pix_cost[row][col].resize(this->d_range, 0x0000);
      this->sum_cost[row][col].resize(this->d_range, 0x0000);
    }
  }

  // Resize vector for Agg Cost
  this->agg_cost.resize(this->scanpath);
  for (int path = 0; path < this->scanpath; path++) {
    this->agg_cost[path].resize(this->rows);
    for (int row = 0; row < this->rows; row++) {
      this->agg_cost[path][row].resize(this->cols);
      for (int col = 0; col < this->cols; col++) {
        this->agg_cost[path][row][col].resize(this->d_range, 0x0000);
      }
    }
  }

  return;
}

void Sgbm::compute_disp(cv::Mat &left, cv::Mat &right, cv::Mat &disp)
{
  return;
}

void Sgbm::calc_matching_cost()
{
  return;
}


unsigned short Sgbm::aggregate_cost(int row, int col, int depth, int path, cost_3d_array &pix_cost, cost_4d_array &agg_cost) {

  // Depth loop for current pix.
  unsigned long val0 = 0xFFFF;
  unsigned long val1 = 0xFFFF;
  unsigned long val2 = 0xFFFF;
  unsigned long val3 = 0xFFFF;
  unsigned long min_prev_d = 0xFFFF;

  int dcol = this->scanlines.path8[path].dcol;
  int drow = this->scanlines.path8[path].drow;

  // Pixel matching cost for current pix.
  unsigned long indiv_cost = pix_cost[row][col][depth];

  if (row - drow < 0 || this->rows <= row - drow || col - dcol < 0 || this->cols <= col - dcol) {
    agg_cost[path][row][col][depth] = indiv_cost;
    return agg_cost[path][row][col][depth];
  }

  // Depth loop for previous pix.
  for (int dd = 0; dd < this->d_range; dd++) {
    unsigned long prev = agg_cost[path][row-drow][col-dcol][dd];
    if (prev < min_prev_d) {
      min_prev_d = prev;
    }
    
    if (depth == dd) {
      val0 = prev;
    } else if (depth == dd + 1) {
      val1 = prev + this->p1;
    } else if (depth == dd - 1) {
      val2 = prev + this->p1;
    } else {
      unsigned long tmp = prev + this->p2;
      if (tmp < val3) {
        val3 = tmp;
      }            
    }
  }

  // Select minimum cost for current pix.
  agg_cost[path][row][col][depth] = std::min(std::min(std::min(val0, val1), val2), val3) + indiv_cost - min_prev_d;
  //agg_cost[path][row][col][depth] = indiv_cost;

  return agg_cost[path][row][col][depth];
}

void Sgbm::aggregate_cost_for_each_scanline(cost_3d_array &pix_cost, cost_4d_array &agg_cost, cost_3d_array &sum_cost)
{
  // Cost aggregation for positive direction.
  for (int row = 0; row < this->rows; row++) {
    for (int col = 0; col < this->cols; col++) {
      for (int path = 0; path < this->scanlines.path8.size(); path++) {
        if (this->scanlines.path8[path].posdir) {
          //std::cout << "Pos : " << path << std::endl;
          for (int d = 0; d < this->d_range; d++) {
            sum_cost[row][col][d] += aggregate_cost(row, col, d, path, pix_cost, agg_cost);
          }
        }
      }
    }
  }

  // Cost aggregation for negative direction.
  for (int row = this->rows - 1; 0 <= row; row--) {
    for (int col = this->cols - 1; 0 <= col; col--) {
      for (int path = 0; path < this->scanlines.path8.size(); path++) {
        if (!this->scanlines.path8[path].posdir) {
          //std::cout << "Neg : " << path << std::endl;
          for (int d = 0; d < this->d_range; d++) {
            sum_cost[row][col][d] += aggregate_cost(row, col, d, path, pix_cost, agg_cost);
          }
        }
      }
    }
  }
  return;
}

void Sgbm::calc_disparity(cost_3d_array &sum_cost, cv::Mat &disp_img)
{
  for (int row = 0; row < this->rows; row++) {
    for (int col = 0; col < this->cols; col++) {
      unsigned char min_depth = 0;
      unsigned long min_cost = sum_cost[row][col][min_depth];
      for (int d = 1; d < this->d_range; d++) {
        unsigned long tmp_cost = sum_cost[row][col][d];
        if (tmp_cost < min_cost) {
          min_cost = tmp_cost;
          min_depth = d;
        }
      }
      disp_img.at<unsigned char>(row, col) = min_depth;
    } 
  } 

  return;
}

void Sgbm::census_transform(cv::Mat &img, cv::Mat &census)
{
  unsigned char * const img_pnt_st = img.data;
  unsigned char * const census_pnt_st = census.data;

  for (int row=1; row<rows-1; row++) {
    for (int col=1; col<cols-1; col++) {

      unsigned char *center_pnt = img_pnt_st + cols*row + col;
      unsigned char val = 0;
      for (int drow=-1; drow<=1; drow++) {
        for (int dcol=-1; dcol<=1; dcol++) {
          
          if (drow == 0 && dcol == 0) {
            continue;
          }
          unsigned char tmp = *(center_pnt + dcol + drow*cols);
          val = (val + (tmp < *center_pnt ? 0 : 1)) << 1;        
        }
      }
      *(census_pnt_st + cols*row + col) = val;
    }
  }
  return;
}

void Sgbm::calc_pixel_cost(cv::Mat &census_l, cv::Mat &census_r, cost_3d_array &pix_cost) {

  unsigned char * const census_l_ptr_st = census_l.data;
  unsigned char * const census_r_ptr_st = census_r.data;

  for (int row = 0; row < this->rows; row++) {
    for (int col = 0; col < this->cols; col++) {
      unsigned char val_l = static_cast<unsigned char>(*(census_l_ptr_st + row*cols + col));
      for (int d = 0; d < this->d_range; d++) {
        unsigned char val_r = 0;
        if (col - d >= 0) {
          val_r = static_cast<unsigned char>(*(census_r_ptr_st + row*cols + col - d));
        }
        pix_cost[row][col][d] = calc_hamming_dist(val_l, val_r);
      }
    }
  }
}

unsigned char Sgbm::calc_hamming_dist(unsigned char val_l, unsigned char val_r) {

  unsigned char dist = 0;
  unsigned char d = val_l ^ val_r;

  while(d) {
    d = d & (d - 1);
    dist++;
  }
  return dist;  
}

