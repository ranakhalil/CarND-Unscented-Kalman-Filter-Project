#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

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
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

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
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

	if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
		/**
		Convert radar from polar to cartesian coordinates and initialize state.
		*/
	}
	else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
		/**
		Initialize state.
		*/
	}
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  //set state dimension
	int n_x = 5;

	//set augmented dimension
	int n_aug = 7;

	//set measurement dimension, radar can measure r, phi, and r_dot
	int n_z = 3;

	//define spreading parameter
	double lambda = 3 - n_aug;

	//set vector for weights
	VectorXd weights = VectorXd(2 * n_aug + 1);
	double weight_0 = lambda / (lambda + n_aug);
	weights(0) = weight_0;
	for (int i = 1; i<2 * n_aug + 1; i++) {
		double weight = 0.5 / (n_aug + lambda);
		weights(i) = weight;
	}

	//radar measurement noise standard deviation radius in m
	double std_radr = 0.3;

	//radar measurement noise standard deviation angle in rad
	double std_radphi = 0.0175;

	//radar measurement noise standard deviation radius change in m/s
	double std_radrd = 0.1;

	//create example matrix with predicted sigma points
	MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);
	
	//create matrix for sigma points in measurement space
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug + 1);

	//transform sigma points into measurement space
	for (int i = 0; i < 2 * n_aug + 1; i++) {  //2n+1 simga points

											   // extract values for better readibility
		double p_x = Xsig_pred(0, i);
		double p_y = Xsig_pred(1, i);
		double v = Xsig_pred(2, i);
		double yaw = Xsig_pred(3, i);

		double v1 = cos(yaw)*v;
		double v2 = sin(yaw)*v;

		// measurement model
		Zsig(0, i) = sqrt(p_x*p_x + p_y*p_y);                        //r
		Zsig(1, i) = atan2(p_y, p_x);                                 //phi
		Zsig(2, i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
	}

	//mean predicted measurement
	VectorXd z_pred = VectorXd(n_z);
	z_pred.fill(0.0);
	for (int i = 0; i < 2 * n_aug + 1; i++) {
		z_pred = z_pred + weights(i) * Zsig.col(i);
	}

	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z, n_z);
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug + 1; i++) {  //2n+1 simga points
											   //residual
		VectorXd z_diff = Zsig.col(i) - z_pred;

		//angle normalization
		while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
		while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

		S = S + weights(i) * z_diff * z_diff.transpose();
	}

	//add measurement noise covariance matrix
	MatrixXd R = MatrixXd(n_z, n_z);
	R << std_radr*std_radr, 0, 0,
		0, std_radphi*std_radphi, 0,
		0, 0, std_radrd*std_radrd;
	S = S + R;

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  //set state dimension
	int n_x = 5;

	//set augmented dimension
	int n_aug = 7;

	//set measurement dimension, radar can measure r, phi, and r_dot
	int n_z = 3;

	//define spreading parameter
	double lambda = 3 - n_aug;

	//set vector for weights
	VectorXd weights = VectorXd(2 * n_aug + 1);
	double weight_0 = lambda / (lambda + n_aug);
	weights(0) = weight_0;
	for (int i = 1; i<2 * n_aug + 1; i++) {  //2n+1 weights
		double weight = 0.5 / (n_aug + lambda);
		weights(i) = weight;
	}

	//create example matrix with predicted sigma points
	MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);

	//create example vector for predicted state mean
	VectorXd x = VectorXd(n_x);

	//create example matrix for predicted state covariance
	MatrixXd P = MatrixXd(n_x, n_x);

	//create example matrix with sigma points in measurement space
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug + 1);

	//create example vector for mean predicted measurement
	VectorXd z_pred = VectorXd(n_z);

	//create example matrix for predicted measurement covariance
	MatrixXd S = MatrixXd(n_z, n_z);

	//create example vector for incoming radar measurement
	VectorXd z = VectorXd(n_z);
	z <<
		5.9214,
		0.2187,
		2.0062;

	//create matrix for cross correlation Tc
	MatrixXd Tc = MatrixXd(n_x, n_z);

	//calculate cross correlation matrix
	Tc.fill(0.0);
	for (int i = 0; i < 2 * n_aug + 1; i++) {  //2n+1 simga points

											   //residual
		VectorXd z_diff = Zsig.col(i) - z_pred;
		//angle normalization
		while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
		while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

		// state difference
		VectorXd x_diff = Xsig_pred.col(i) - x;
		//angle normalization
		while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;
		while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;

		Tc = Tc + weights(i) * x_diff * z_diff.transpose();
	}

	//Kalman gain K;
	MatrixXd K = Tc * S.inverse();

	//residual
	VectorXd z_diff = z - z_pred;

	//angle normalization
	while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
	while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;

	//update state mean and covariance matrix
	x = x + K * z_diff;
	P = P - K*S*K.transpose();
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}

void UKF::GenerateSigmaPoints(MatrixXd* Xsig_out) {

	//set state dimension
	int n_x = 5;

	//define spreading parameter
	double lambda = 3 - n_x;

	//set example state
	VectorXd x = VectorXd(n_x);

	//set example covariance matrix
	MatrixXd P = MatrixXd(n_x, n_x);

	//create sigma point matrix
	// X k|k
	MatrixXd Xsig = MatrixXd(n_x, 2 * n_x + 1);

	//calculate square root of P
	MatrixXd A = P.llt().matrixL();

	//set first column of sigma point matrix
	Xsig.col(0) = x;

	//set remaining sigma points
	for (int i = 0; i < n_x; i++)
	{
		Xsig.col(i + 1) = x + sqrt(lambda + n_x) * A.col(i);
		Xsig.col(i + 1 + n_x) = x - sqrt(lambda + n_x) * A.col(i);
	}
}


void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) {

	//set state dimension
	int n_x = 5;

	//set augmented dimension
	int n_aug = 7;

	//Process noise standard deviation longitudinal acceleration in m/s^2
	double std_a = 0.2;

	//Process noise standard deviation yaw acceleration in rad/s^2
	double std_yawdd = 0.2;

	//define spreading parameter
	double lambda = 3 - n_aug;

	//set example state
	VectorXd x = VectorXd(n_x);

	//create example covariance matrix
	MatrixXd P = MatrixXd(n_x, n_x);

	//create augmented mean vector
	VectorXd x_aug = VectorXd(7);

	//create augmented state covariance
	MatrixXd P_aug = MatrixXd(7, 7);

	//create sigma point matrix
	MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);

	//create augmented mean state
	x_aug.head(5) = x;
	x_aug(5) = 0;
	x_aug(6) = 0;

	//create augmented covariance matrix
	P_aug.fill(0.0);
	P_aug.topLeftCorner(5, 5) = P;
	P_aug(5, 5) = std_a*std_a;
	P_aug(6, 6) = std_yawdd*std_yawdd;

	//create square root matrix
	MatrixXd L = P_aug.llt().matrixL();

	//create augmented sigma points
	Xsig_aug.col(0) = x_aug;
	for (int i = 0; i < n_aug; i++)
	{
		Xsig_aug.col(i + 1) = x_aug + sqrt(lambda + n_aug) * L.col(i);
		Xsig_aug.col(i + 1 + n_aug) = x_aug - sqrt(lambda + n_aug) * L.col(i);
	}

}

void UKF::SigmaPointPrediction(MatrixXd* Xsig_out) {

	//set state dimension
	int n_x = 5;

	//set augmented dimension
	int n_aug = 7;

	//create example sigma point matrix
	MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);

	//create matrix with predicted sigma points as columns
	MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);

	double delta_t = 0.1; //time diff in sec

	//predict sigma points

	for (int i = 0; i < 2 * n_aug + 1; i++)
	{
		//extract values for better readability
		double p_x = Xsig_aug(0, i);
		double p_y = Xsig_aug(1, i);
		double v = Xsig_aug(2, i);
		double yaw = Xsig_aug(3, i);
		double yawd = Xsig_aug(4, i);
		double nu_a = Xsig_aug(5, i);
		double nu_yawdd = Xsig_aug(6, i);

		//predicted state values
		double px_p, py_p;

		//avoid division by zero
		if (fabs(yawd) > 0.001) {
			px_p = p_x + v / yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
			py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd*delta_t));
		}
		else {
			px_p = p_x + v*delta_t*cos(yaw);
			py_p = p_y + v*delta_t*sin(yaw);
		}

		double v_p = v;
		double yaw_p = yaw + yawd*delta_t;
		double yawd_p = yawd;

		//add noise
		px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
		py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
		v_p = v_p + nu_a*delta_t;

		yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
		yawd_p = yawd_p + nu_yawdd*delta_t;

		//write predicted sigma point into right column
		Xsig_pred(0, i) = px_p;
		Xsig_pred(1, i) = py_p;
		Xsig_pred(2, i) = v_p;
		Xsig_pred(3, i) = yaw_p;
		Xsig_pred(4, i) = yawd_p;
	}
}
