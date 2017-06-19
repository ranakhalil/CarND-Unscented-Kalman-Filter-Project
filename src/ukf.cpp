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

static double EPS = 0.001;

UKF::UKF() {
	// if this is false, laser measurements will be ignored (except during init)
	use_laser_ = true;

	// if this is false, radar measurements will be ignored (except during init)
	use_radar_ = true;

	// state dimension
	n_x_ = 5;

	n_z_lidar = 2;

	n_z_radar = 3;

	// initial state vector
	x_ = VectorXd(n_x_);

	// initial covariance matrix
	P_ = MatrixXd(n_x_, n_x_);

	// Process noise standard deviation longitudinal acceleration in m/s^2
	std_a_ = 5;

	// Process noise standard deviation yaw acceleration in rad/s^2
	std_yawdd_ = 0.8 * M_PI;

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
	delta_t = 0.0;

	H_laser_ = MatrixXd(2, 5);
	H_laser_ << 1, 0, 0, 0, 0,
		       0, 1, 0, 0, 0;

	//add measurement noise covariance matrix for lidar
	R_laser_ = MatrixXd(n_z_lidar, n_z_lidar);

	R_laser_ << std_laspx_*std_laspx_, 0,
		0, std_laspy_*std_laspy_;

	R_radar_ = MatrixXd(n_z_radar, n_z_radar);
	//add measurement noise covariance matrix for radar
	R_radar_ << std_radr_ * std_radr_, 0, 0,
		0, std_radphi_*std_radphi_, 0,
		0, 0, std_radrd_ * std_radrd_;
}

UKF::~UKF() {}


void UKF::normalizeAngle(double& angle)
{
	if (angle < EPS)
	{
		angle = atan2(sin(EPS), cos(EPS));
	}
	else
	{
		angle = atan2(sin(angle), cos(angle));
	}
	
}

void UKF::AugmentedSigmaPoints() {
	
	x_aug = VectorXd(n_aug_);
	Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
	P_aug = MatrixXd(n_aug_, n_aug_);

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
						  
	Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
	for (int i = 0; i< 2 * n_aug_ + 1; i++)
	{
		//extract values for better readability
		const double p_x = Xsig_aug(0, i);
		const double p_y = Xsig_aug(1, i);
		const double v = Xsig_aug(2, i);
		const double yaw = Xsig_aug(3, i);
		const double yawd = Xsig_aug(4, i);
		const double nu_a = Xsig_aug(5, i);
		const double nu_yawdd = Xsig_aug(6, i);

		//predicted state values
		double px_p, py_p;

		//avoid division by zero
		if (fabs(yawd) > EPS) {
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
		px_p += 0.5*nu_a*delta_t*delta_t * cos(yaw);
		py_p += 0.5*nu_a*delta_t*delta_t * sin(yaw);
		v_p = v_p + nu_a*delta_t;

		yaw_p += 0.5 * nu_yawdd*delta_t*delta_t;
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

	//predicted state mean
	x_.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
		x_ += weights_(i) * Xsig_pred_.col(i);
	}

	//predicted state covariance matrix
	P_.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		//angle normalization
		normalizeAngle(x_diff(3));
		P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
	}
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
		//cout << "P_" << P_ << endl;

		if (meas_package.sensor_type_ == MeasurementPackage::LASER)
		{
			x_ << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), 0.0, 0.0, 0.0;

			if (x_(0) < EPS && x_(1) < EPS)
			{
				x_(0) = EPS;
				x_(1) = EPS;
			}
		}
		else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
		{
			const float rho = meas_package.raw_measurements_(0);
			const float phi = meas_package.raw_measurements_(1);
			const float rho_dot = meas_package.raw_measurements_(2);

			const float px = rho * cos(phi);
			const float py = rho * sin(phi);
			//float vx = rho_dot * cos(phi);
			//float vy = rho_dot * sin(phi);
			// sqrt(ro_dot^2 * (cos^2(phi) + sin^2(phi)) or sqrt(cos^2(phi) * ro_dot^2 + sin^2(phi) * ro_dot^2)
			const float velocity = sqrt(pow(rho_dot, 2) * (pow(cos(phi), 2) + pow(sin(phi), 2)));
			x_ << px, py, velocity, 0.0, 0.0;
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
		//cout << "Update Lidar " << endl;
		UpdateLidar(meas_package);
	}
	else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
	{
		//cout << "Update Radar " << endl;
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

	// Augment Sigma points
	AugmentedSigmaPoints();

	// Sigma point prediction
	SigmaPointPrediction();
	
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
	z = VectorXd(3);
	z.fill(0.0);

	z = meas_package.raw_measurements_;

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
	z = meas_package.raw_measurements_;
	//set vector for weights
	Zsig = MatrixXd(3, 2 * n_aug_ + 1);
	Zsig.fill(0.0);

	//transform sigma points into measurement space
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

											   // extract values for better readibility
		const double p_x = Xsig_pred_(0, i);
		const double p_y = Xsig_pred_(1, i);
		const double v = Xsig_pred_(2, i);
		const double yaw = Xsig_pred_(3, i);

		const double v1 = cos(yaw)*v;
		const double v2 = sin(yaw)*v;

		// measurement model
		Zsig(0, i) = sqrt(p_x*p_x + p_y*p_y);                                       //r
		Zsig(1, i) = atan2(max(EPS, p_y), max(EPS, p_x));                                    //phi
		Zsig(2, i) = (p_x*v1 + p_y*v2) / max(EPS, sqrt(p_x * p_x + p_y * p_y));   //r_dot
	}

	//mean predicted measurement
	VectorXd z_pred = VectorXd(n_z_radar);
	z_pred.fill(0.0);

	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	}

	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z_radar, n_z_radar);
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
											   //residual
		VectorXd z_diff = Zsig.col(i) - z_pred;
		//angle normalization
		normalizeAngle(z_diff(1));
		S = S + weights_(i) * z_diff * z_diff.transpose();
	}


	S = S + R_radar_;

	// Update Radar:
	//calculate cross correlation matrix
	Tc = MatrixXd(n_x_, 3);
	Tc.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

											   //residual
		VectorXd z_diff = Zsig.col(i) - z_pred;
		//angle normalization
		normalizeAngle(z_diff(1));

		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		//angle normalization
		normalizeAngle(x_diff(3));

		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}

	//Kalman gain K;
	MatrixXd K = Tc * S.inverse();

	//residual
	VectorXd z_diff = z - z_pred;

	//angle normalization
	normalizeAngle(z_diff(1));

	//update state mean and covariance matrix
	x_ = x_ + K * z_diff;
	P_ = P_ - K * S * K.transpose();

	NIS_radar_ = z.transpose() * S.inverse() * z;
}