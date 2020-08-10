#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30; //play with this

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30; //play with this
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  is_initialized_ = false;
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;
  weights_ = VectorXd(2*n_aug_ + 1);
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);
  time_us_ = 0.0;

  R_lidar_ = MatrixXd(2,2);
  R_lidar_ << std_laspx_*std_laspx_, 0, 0, std_laspy_*std_laspy_;
  R_radar_ = MatrixXd(3,3);
  R_radar_ << std_radrd_*std_radrd_, 0, 0, 0, std_radphi_*std_radphi_, 0, 0, 0, std_radrd_*std_radrd_;
  Q_ = MatrixXd(2,2);
  Q_ << std_a_*std_a_, 0, 0, std_yawdd_*std_yawdd_;
  H_lidar_ = MatrixXd(2,5);
  H_lidar_ << 1, 0, 0, 0, 0, 0, 1, 0, 0, 0;

}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if (is_initialized_ == false) {
    // initialize the state the initial sensor measurements
    // initialize the state covariance matrix

    std::cout << "Step1: initializing the UKF" << std::endl;

    // initialize state = [px, py, v, phi, phid ]
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // Lidar directly gives [x, y, z]
      x_ << meas_package.raw_measurements_[0],
            meas_package.raw_measurements_[1],
            0.0, 0.0, 0.0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // radar gives polar coordinates [rho, phi, rhod]
      double rho = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      x_ << rho * cos(phi),
            rho*sin(phi),
            0.0, 0.0, 0.0;
    }

    // starting with all 1s on the diagonal
    MatrixXd diagonal = MatrixXd(5,1);
    diagonal << 1.0, 1.0, 1.0, 1.0, 1.0;
    P_.fill(0.0);
    P_ = diagonal.matrix().asDiagonal();
    time_us_ = meas_package.timestamp_;

    is_initialized_ = true;
    return;
  }
  
  // current measurement - last measurement
  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;
  std::cout << "timestamp: " << meas_package.timestamp_ << std::endl;
  Prediction(delta_t);
  // already initialized, run regular UKF
  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    std::cout << "DEBUG: LIDAR CALL" << std::endl;
    UpdateLidar(meas_package);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    std::cout << "DEBUG: RADAR CALL" << std::endl;
    UpdateRadar(meas_package);
  }
  else {
    // Wrong sensor type
  }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

  // 1. Generate augmented sigma points -> Xsig_aug
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  x_aug.head(n_x_) = x_;
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  MatrixXd Q(2,2);
  Q << std_a_*std_a_, 0, 0, std_yawdd_*std_yawdd_;
  P_aug.bottomRightCorner(2,2) = Q;
  MatrixXd A = P_aug.llt().matrixL();
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug.col(i+1)     = x_aug + sqrt(lambda_+n_aug_) * A.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * A.col(i);
  }
  // 2. Propagate the sigma points -> Xsig_pred_
  float x1, x2, x3, x4, x5, x6, x7;
  float px1, px2, px3, px4, px5;
  for (int j=0; j < Xsig_aug.cols(); j++) {
      VectorXd sigma = Xsig_aug.col(j);
      x1 = sigma(0); //px
      x2 = sigma(1); //py
      x3 = sigma(2); //v
      x4 = sigma(3); //phi
      x5 = sigma(4); //phi_dot
      x6 = sigma(5); //nu_a
      x7 = sigma(6); //nu_phidotdot
      if (fabs(x5) < 0.00001) {
          //divide by zero
          px1 = x1 + x3*cos(x4)*delta_t + 0.5*(delta_t*delta_t)*cos(x4)*x6;
          px2 = x2 + x3*sin(x4)*delta_t + 0.5*(delta_t*delta_t)*sin(x4)*x6;
      }
      else {
          px1 = x1 + (x3/x5)*(sin(x4+x5*delta_t) - sin(x4)) + 0.5*(delta_t*delta_t)*cos(x4)*x6;
          px2 = x2 + (x3/x5)*(-cos(x4+x5*delta_t) + cos(x4)) + 0.5*(delta_t*delta_t)*sin(x4)*x6;
      }
      px3 = x3 + 0 + delta_t*x6;
      px4 = x4 + x5*delta_t + 0.5*(delta_t*delta_t)*x7;
      px5 = x5 + 0 + delta_t*x7;
      
      Xsig_pred_.col(j) << px1,px2,px3,px4,px5;
  }
  // 4. Predict the mean, covariance
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i=1; i < weights_.size(); i++) {
      weights_(i) = 1 / (2*(lambda_ + n_aug_));
  }
  // predict state mean
  for (int i=0; i < 2*n_aug_+1; i++) {
      x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }
  for (int i=0; i < 2*n_aug_+1; i++) {
      VectorXd center = Xsig_pred_.col(i) - x_;
      center(3) = Normalize(center(3));
      MatrixXd outer = center * center.transpose();
      P_ += weights_(i) * outer;
  }

  std::cout << "Predicted state" << std::endl;
  std::cout << x_ << std::endl;
  std::cout << "Predicted covariance matrix" << std::endl;
  std::cout << P_ << std::endl;
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  // 1. Calculate the difference between predicted and actual
  VectorXd z = VectorXd(2);
  z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1];
  VectorXd z_pred = H_lidar_ * x_;
  VectorXd y = z - z_pred;

  // 2. Calculate the Kalman gain
  MatrixXd S = H_lidar_ * P_ * H_lidar_.transpose() + R_lidar_;
  MatrixXd K = P_ * H_lidar_.transpose() * S.inverse();

  // 3. Update the state and covariance
  x_ = x_ + K * y;
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K*H_lidar_) * P_;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  int n_z = 3;
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  VectorXd z_pred = VectorXd(n_z);
  MatrixXd S = MatrixXd(n_z,n_z);

  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // extract values for better readability
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                       // r
    Zsig(1,i) = atan2(p_y,p_x);                                // phi
    Zsig(2,i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);   // r_dot
  }

  for (int i=0; i < 2*n_aug_+1; ++i) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // calculate innovation covariance matrix S
  MatrixXd R = MatrixXd(n_z,n_z);
  R << std_radr_*std_radr_, 0, 0,
       0,std_radphi_*std_radphi_, 0,
       0, 0, std_radrd_*std_radrd_;
  for (int j=0; j < 2*n_aug_+1; j++) {
      VectorXd center = Zsig.col(j) - z_pred;
      center(1) = Normalize(center(1));
      S = S + weights_(j) * center * center.transpose();
  }
  S = S + R;

  // update state
  VectorXd z = VectorXd(n_z);
  double rho = meas_package.raw_measurements_[0];
  double phi = meas_package.raw_measurements_[1];
  //z << rho * cos(phi), rho*sin(phi), 0.0;
  z << rho, phi, 0.0;
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  // calculate cross correlation matrix
  for (int j=0; j < 2*n_aug_ +1; j++) {
      VectorXd x_center = Xsig_pred_.col(j) - x_;
      VectorXd z_center = Zsig.col(j) - z_pred;
      // normalize the angles
      Tc = Tc + weights_(j) * x_center * z_center.transpose();
  }

  // calculate Kalman gain K;
  MatrixXd K;
  K = Tc * S.inverse();
  
  // update state mean and covariance matrix
  VectorXd zdiff = z - z_pred;
  zdiff(1) = Normalize(zdiff(1));
  x_ = x_ + K * zdiff; // normalize the angles
  P_ = P_ - K*S*K.transpose();
}

double UKF::Normalize(double angle) {
  while (angle> M_PI)
    angle-=2.*M_PI;
  while (angle<-M_PI)
    angle+=2.*M_PI;
  return angle;
}