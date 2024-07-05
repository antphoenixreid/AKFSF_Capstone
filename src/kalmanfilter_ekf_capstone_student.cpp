// ------------------------------------------------------------------------------- //
// Advanced Kalman Filtering and Sensor Fusion Course - Extended Kalman Filter
//
// ####### STUDENT FILE #######
//
// Usage:
// -Rename this file to "kalmanfilter.cpp" if you want to use this code.

#include "kalmanfilter.h"
#include "utils.h"

// -------------------------------------------------- //
// YOU CAN USE AND MODIFY THESE CONSTANTS HERE
constexpr double ACCEL_STD = 1.0;
constexpr double GYRO_STD = 0.01/180.0 * M_PI;
constexpr double GYRO_BIAS_STD = 0.005/180.0*M_PI;
constexpr double INIT_VEL_STD = 10.0;
constexpr double INIT_PSI_STD = 45.0/180.0 * M_PI;
constexpr double GPS_POS_STD = 3.0;
constexpr double LIDAR_RANGE_STD = 3.0;
constexpr double LIDAR_THETA_STD = 0.02;
// -------------------------------------------------- //

void KalmanFilter::handleLidarMeasurements(const std::vector<LidarMeasurement>& dataset, const BeaconMap& map)
{
    // Assume No Correlation between the Measurements and Update Sequentially
    for(const auto& meas : dataset) {handleLidarMeasurement(meas, map);}
}

void KalmanFilter::handleLidarMeasurement(LidarMeasurement meas, const BeaconMap& map)
{
    if (isInitialised())
    {
        VectorXd state = getState();
        MatrixXd cov = getCovariance();

        // Implement The Kalman Filter Update Step for the Lidar Measurements in the 
        // section below.
        // HINT: use the wrapAngle() function on angular values to always keep angle
        // values within correct range, otherwise strange angle effects might be seen.
        // HINT: You can use the constants: LIDAR_RANGE_STD, LIDAR_THETA_STD
        // HINT: The mapped-matched beacon position can be accessed by the variables
        // map_beacon.x and map_beacon.y
        // ----------------------------------------------------------------------- //
        // ENTER YOUR CODE HERE

        BeaconData map_beacon = map.getBeaconWithId(meas.id); // Match Beacon with built in Data Association Id
        if (meas.id != -1 && map_beacon.id != -1)
        {           
            // The map matched beacon positions can be accessed using: map_beacon.x AND map_beacon.y

            // Measurement Vector
            VectorXd z = Vector2d::Zero();
            z << meas.range, meas.theta;

            // Implement Lidar Measurement Model
            VectorXd z_hat = Vector2d::Zero();
            double delta_x = map_beacon.x - state[0];
            double delta_y = map_beacon.y - state[1];
            double zhat_range = sqrt(delta_x*delta_x + delta_y*delta_y);
            double zhat_theta = wrapAngle(atan2(delta_y,delta_x) - state[2]);
            z_hat << zhat_range, zhat_theta;

            // Implement the Innovation Calculations
            VectorXd y = z - z_hat;
            y(1) = wrapAngle(y(1)); 

            MatrixXd H = MatrixXd(2,5);
            H << -delta_x/zhat_range,-delta_y/zhat_range,0,0,0,delta_y/zhat_range/zhat_range,-delta_x/zhat_range/zhat_range,-1,0,0;

            MatrixXd R = Matrix2d::Zero();
            R(0,0) = LIDAR_RANGE_STD*LIDAR_RANGE_STD;
            R(1,1) = LIDAR_THETA_STD*LIDAR_THETA_STD;

            MatrixXd S = H*cov*H.transpose() + R;

            // EKF Update Step Equations
            MatrixXd K = cov*H.transpose()*S.inverse();

            state = state + K*y;
            cov = (MatrixXd::Identity(5,5) - K*H)*cov;
        }
        

        // ----------------------------------------------------------------------- //

        setState(state);
        setCovariance(cov);
    }
    else
    {
        // Data Association (LIDAR)
        BeaconData map_beacon = map.getBeaconWithId(meas.id); // temp; TODO: Data Association without built-in method

        if (meas.id != -1 && map_beacon.id != -1)
        {
            if (m_init_position_valid)
            {
                double delta_x = map_beacon.x - m_init_position_x;
                double delta_y = map_beacon.y - m_init_position_y;

                // Check if moved enough
                if ((delta_x*delta_x + delta_y*delta_y) > 3*GPS_POS_STD)
                {
                    // Estimate heading from delta position
                    m_init_heading = wrapAngle(atan2(delta_y, delta_x) - meas.theta);
                    m_init_heading_valid = true;
                }
            }
        }
    }
}

void KalmanFilter::predictionStep(GyroMeasurement gyro, double dt)
{
    if (isInitialised())
    {
        VectorXd state = getState();
        MatrixXd cov = getCovariance();

        // Implement The Kalman Filter Prediction Step for the system in the  
        // section below.
        // HINT: Assume the state vector has the form [PX, PY, PSI, V].
        // HINT: Use the Gyroscope measurement as an input into the prediction step.
        // HINT: You can use the constants: ACCEL_STD, GYRO_STD
        // HINT: use the wrapAngle() function on angular values to always keep angle
        // values within correct range, otherwise strange angle effects might be seen.
        // ----------------------------------------------------------------------- //
        // ENTER YOUR CODE HERE

        // Update State
        double p_x = state(0);
        double p_y = state(1);
        double psi = state(2);
        double vel = state(3);
        double bias = state(4);

        double d_px = dt*(vel*cos(psi));
        double d_py = dt*(vel*sin(psi));
        double d_psi = dt*(gyro.psi_dot - bias);

        state << p_x + d_px, p_y + d_py, wrapAngle(psi + d_psi), vel, bias;

        // Update Covariance
        MatrixXd F = MatrixXd::Identity(5,5);
        F(0,2) = -dt*vel*sin(psi);
        F(0,3) = dt*cos(psi);
        F(1,2) = dt*vel*cos(psi);
        F(1,3) = dt*sin(psi);
        F(2,4) = -dt;

        MatrixXd Q = MatrixXd::Zero(5,5);
        Q(2,2) = (dt*dt)*(GYRO_STD*GYRO_STD);
        Q(3,3) = (dt*dt)*(ACCEL_STD*ACCEL_STD);
        Q(4,4) = (dt*dt)*(GYRO_BIAS_STD*GYRO_BIAS_STD);

        cov = F*cov*F.transpose() + Q;

        // ----------------------------------------------------------------------- //

        setState(state);
        setCovariance(cov);
    }
    else
    {
        // Assume Driving Straight and/or Stationary
        m_init_bias = gyro.psi_dot;
        m_init_bias_valid = true;
    } 
}

void KalmanFilter::handleGPSMeasurement(GPSMeasurement meas)
{
    // All this code is the same as the LKF as the measurement model is linear
    // so the EKF update state would just produce the same result.
    if(isInitialised())
    {
        VectorXd state = getState();
        MatrixXd cov = getCovariance();

        VectorXd z = Vector2d::Zero();
        MatrixXd H = MatrixXd(2,5);
        MatrixXd R = Matrix2d::Zero();

        z << meas.x,meas.y;
        H << 1,0,0,0,0,0,1,0,0,0;
        R(0,0) = GPS_POS_STD*GPS_POS_STD;
        R(1,1) = GPS_POS_STD*GPS_POS_STD;

        VectorXd z_hat = H * state;
        VectorXd y = z - z_hat;
        MatrixXd S = H * cov * H.transpose() + R;
        MatrixXd K = cov*H.transpose()*S.inverse();

        VectorXd NIS = y.transpose()*S.inverse()*y;
        if (NIS(0) < 5.99)
        {
            state = state + K*y;
            cov = (MatrixXd::Identity(5,5) - K*H) * cov;
        }

        setState(state);
        setCovariance(cov);
    }
    else
    {
        // You may modify this initialisation routine if you can think of a more
        // robust and accuracy way of initialising the filter.
        // ----------------------------------------------------------------------- //
        // YOU ARE FREE TO MODIFY THE FOLLOWING CODE HERE

        // Set initial position (first measurement)
        if(!m_init_position_valid)
        {
            m_init_position_x = meas.x;
            m_init_position_y = meas.y;
            m_init_position_valid = true;
        }

        // Velocity Assuption/Initiation
        m_init_velocity = 0.5;
        m_init_velocity_valid = true;

        // Estimating Heading from GPS measurement
        if(m_init_position_valid && !m_init_heading_valid)
        {
            double delta_x = meas.x - m_init_position_x;
            double delta_y = meas.y - m_init_position_y;

            // Check if moved enough
            if ((delta_x*delta_x + delta_y*delta_y) > 3*GPS_POS_STD)
            {
                // Estimate heading from delta position
                m_init_heading = wrapAngle(atan2(delta_y, delta_x));
                m_init_heading_valid = true;
            }
        }

        if (m_init_position_valid && m_init_heading_valid && m_init_velocity_valid && m_init_bias_valid)
        {
            VectorXd state = VectorXd::Zero(5);
            MatrixXd cov = MatrixXd::Zero(5,5);

            state(0) = m_init_position_x;
            state(1) = m_init_position_y;
            state(2) = m_init_heading;
            state(3) = m_init_velocity;
            state(4) = m_init_bias;

            cov(0,0) = GPS_POS_STD*GPS_POS_STD;
            cov(1,1) = GPS_POS_STD*GPS_POS_STD;
            cov(2,2) = LIDAR_THETA_STD*LIDAR_THETA_STD;
            cov(3,3) = INIT_VEL_STD*INIT_VEL_STD;
            cov(4,4) = GYRO_STD*GYRO_STD;

            setState(state);
            setCovariance(cov);
        }

        // ----------------------------------------------------------------------- //
    } 
             
}

Matrix2d KalmanFilter::getVehicleStatePositionCovariance()
{
    Matrix2d pos_cov = Matrix2d::Zero();
    MatrixXd cov = getCovariance();
    if (isInitialised() && cov.size() != 0){pos_cov << cov(0,0), cov(0,1), cov(1,0), cov(1,1);}
    return pos_cov;
}

VehicleState KalmanFilter::getVehicleState()
{
    if (isInitialised())
    {
        VectorXd state = getState(); // STATE VECTOR [X,Y,PSI,V,...]
        return VehicleState(state[0],state[1],state[2],state[3]);
    }
    return VehicleState();
}

void KalmanFilter::predictionStep(double dt){}
