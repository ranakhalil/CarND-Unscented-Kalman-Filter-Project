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

	// state dimension
	n_x_ = 5;

	n_z = 3;

	// initial state vector
	x_ = VectorXd(n_x_);

	// initial covariance matrix
	P_ = MatrixXd(n_x_, n_x_);

	// Process noise standard deviation longitudinal acceleration in m/s^2
	std_a_ = 15;

	// Process noise standard deviation yaw acceleration in rad/s^2
	std_yawdd_ = 15;

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

	// Augmented state dimension
	n_aug_ = 7;

	//Sigma points spreading parameter
	lambda_ = 3 - n_aug_;
	
	// Weights of Sigma points 
	weights_ = VectorXd(2 * n_aug_ + 1);

	is_initialized_ = false;

	time_us_ = 0.0;

	Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);

	//create sigma point matrix
	Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

	x_aug = VectorXd(n_aug_);

	P_aug = MatrixXd(n_aug_, n_aug_);

	Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

	delta_t = 0.0;

	Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

	S = MatrixXd(n_z, n_z);

	Tc = MatrixXd(n_x_, n_z);

	z_pred = VectorXd(n_z);
	
	z = VectorXd(n_z);

	H_laser_ = MatrixXd(2, 4);
	H_laser_ << 1, 0, 0, 0,
		       0, 1, 0, 0;

	//add measurement noise covariance matrix for lidar
	R_laser_ = MatrixXd(2, 2);
	R_laser_ << std_laspx_*std_laspx_, 0,
		0, std_laspy_*std_laspy_;

	R_radar_ = MatrixXd(n_z, n_z);

	//add measurement noise covariance matrix for radar
	R_radar_ << std_radr_ * std_radr_, 0, 0,
		0, std_radphi_*std_radphi_, 0,
		0, 0, std_radrd_ * std_radrd_;
}

UKF::~UKF() {}


double UKF::normalizeAngle(double angle)
{
	double normalized_angle = angle;
	while (normalized_angle > M_PI) normalized_angle -= 2.*M_PI;
	while (normalized_angle < -M_PI) normalized_angle += 2.*M_PI;

	return normalized_angle;
}

void UKF::GenerateSigmaPoints() {

	//calculate square root of P
	MatrixXd A = P_.llt().matrixL();

	//set first column of sigma point matrix
	Xsig.col(0) = x_;

	//set remaining sigma points
	for (int i = 0; i < n_x_; i++)
	{
		Xsig.col(i + 1) = x_ + sqrt(lambda_ + n_x_) * A.col(i);
		Xsig.col(i + 1 + n_x_) = x_ - sqrt(lambda_ + n_x_) * A.col(i);
	}
}

void UKF::AugmentedSigmaPoints() {
	
	//create augmented mean state
	x_aug.head(5) = x_;

	//create augmented covariance matrix
	P_aug.fill(0.0);
	P_aug.topLeftCorner(5, 5) = P_;
	P_aug(5, 5) = std_a_ * std_a_;
	P_aug(6, 6) = std_yawdd_ * std_yawdd_;

	//create square root matrix
	MatrixXd L = P_aug.llt().matrixL();

	//create augmented sigma points
	Xsig_aug.col(0) = x_aug;

	for (int i = 0; i< n_aug_; i++)
	{
		Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
		Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
	}
}

void UKF::SigmaPointPrediction() {
						  
	for (int i = 0; i< 2 * n_aug_ + 1; i++)
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

		yaw_p = yaw_p + 0.5 * nu_yawdd*delta_t*delta_t;
		yawd_p = yawd_p + nu_yawdd*delta_t;

		//write predicted sigma point into right column
		Xsig_pred_(0, i) = px_p;
		Xsig_pred_(1, i) = py_p;
		Xsig_pred_(2, i) = v_p;
		Xsig_pred_(3, i) = yaw_p;
		Xsig_pred_(4, i) = yawd_p;
	}
}

void UKF::PredictMeanAndCovariance() {
	// set weights
	double weight_0 = lambda_ / (lambda_ + n_aug_);
	
	weights_(0) = weight_0;
	for (int i = 1; i<2 * n_aug_ + 1; i++) {  //2n+1 weights
		double weight = 0.5 / (n_aug_ + lambda_);
		weights_(i) = weight;
	}

	cout << "Weights" << weights_ << endl;

	//predicted state mean
	x_.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
		x_ = x_ + weights_(i) * Xsig_pred_.col(i);
	}

	//predicted state covariance matrix
	P_.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		//angle normalization
		x_diff(3) = normalizeAngle(x_diff(3));
		P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
	}

	cout << "P_" << P_ << endl;
}

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

	if (!is_initialized_)
	{
		// Set the predicted sigma points matrix as an identity matrix
		P_.setIdentity();

		cout << "P_" << P_ << endl;

		if (meas_package.sensor_type_ == MeasurementPackage::LASER)
		{
			x_ << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), 0.0, 0.0, 0.0;

			if (x_(0) < 0.0001 && x_(1) < 0.0001)
			{
				x_(0) = 0.0001;
				x_(1) = 0.0001;
			}
		}
		else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
		{
			float rho = meas_package.raw_measurements_(0);
			float phi = meas_package.raw_measurements_(1);
			float rho_dot = meas_package.raw_measurements_(2);

			float px = rho * cos(phi);
			float py = rho * sin(phi);
			float vx = rho_dot * cos(phi);
			float vy = rho_dot * sin(phi);

			x_ << px, py, sqrt(vx * vx + vy * vy), 0.0, 0.0;
		}
		time_us_ = meas_package.timestamp_;

		// Initialize weights 
		double weight_0 = lambda_ / (lambda_ + n_aug_);
		weights_(0) = weight_0;
		for (int i = 1; i<2 * n_aug_ + 1; i++) {
			weights_(i) = 0.5 / (n_aug_ + lambda_);
		}

		is_initialized_ = true;
		return;
	}
	

	// learnt this from the forums 
	// https://discussions.udacity.com/t/delta-t-and-time-us-/255521

	delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;

	//Predict:

	Prediction(delta_t);

	//Update:

	if (meas_package.sensor_type_ == MeasurementPackage::LASER)
	{
		cout << "Update Lidar " << endl;
		UpdateLidar(meas_package);
	}
	else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
	{
		cout << "Update Radar " << endl;

		UpdateRadar(meas_package);
	}

	time_us_ = meas_package.timestamp_;
	
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

	// Generate Sigma points
	cout << "Generate Sigma points " << endl;
	GenerateSigmaPoints();

	cout << "Augmented Sigma points " << endl;
	// Augment Sigma points
	AugmentedSigmaPoints();

	cout << "Sigma point prediction " << endl;
	// Sigma point prediction
	SigmaPointPrediction();
	
	cout << "Predicted Mean and Covariance " << endl;
	// Predict mean state and covariance
	PredictMeanAndCovariance();
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

	VectorXd z_pred = H_laser_ * x_;
	VectorXd y = z - z_pred;
	MatrixXd Ht = H_laser_.transpose();
	MatrixXd S = H_laser_ * P_ * Ht + R_laser_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_laser_) * P_;
	NIS_laser_ = z.transpose() * S.inverse() * z;
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

	//set vector for weights
	Zsig.fill(0.0);
	double weight_0 = lambda_ / (lambda_ + n_aug_);
	weights_(0) = weight_0;
	for (int i = 1; i<2 * n_aug_ + 1; i++) {
		double weight = 0.5 / (n_aug_ + lambda_);
		weights_(i) = weight;
	}

	//transform sigma points into measurement space
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

											   // extract values for better readibility
		double p_x = Xsig_pred_(0, i);
		double p_y = Xsig_pred_(1, i);
		double v = Xsig_pred_(2, i);
		double yaw = Xsig_pred_(3, i);

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
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	}

	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z, n_z);
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
											   //residual
		VectorXd z_diff = Zsig.col(i) - z_pred;
		//angle normalization
		z_diff(1) = normalizeAngle(z_diff(1));
		S = S + weights_(i) * z_diff * z_diff.transpose();
	}


	S = S + R_radar_;

	// Update Radar:
	//calculate cross correlation matrix
	Tc.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

											   //residual
		VectorXd z_diff = Zsig.col(i) - z_pred;
		//angle normalization
		z_diff(1) = normalizeAngle(z_diff(1));

		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		//angle normalization
		x_diff(3) = normalizeAngle(x_diff(3));

		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}

	//Kalman gain K;
	MatrixXd K = Tc * S.inverse();

	//residual
	VectorXd z_diff = z - z_pred;

	//angle normalization
	z_diff(1) = normalizeAngle(z_diff(1));

	//update state mean and covariance matrix
	x_ = x_ + K * z_diff;
	P_ = P_ - K * S * K.transpose();

	NIS_radar_ = z.transpose() * S.inverse() * z;
}