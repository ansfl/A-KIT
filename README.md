# A-KIT: Adaptive Kalman-Informed Transformer
## Abstract:
The extended Kalman filter (EKF) is a widely adopted method for sensor fusion in navigation applications. A crucial aspect of the EKF is the online determination of the process noise covariance matrix reflecting the model uncertainty. While common EKF implementation assumes a constant process noise, in real-world scenarios, the process noise varies, leading to inaccuracies in the estimated state and potentially causing the filter to diverge. To cope with such situations, model-based adaptive EKF methods were proposed and demonstrated performance improvements, highlighting the need for a robust adaptive approach. In this paper, we derive and introduce A-KIT, an adaptive Kalman-informed transformer to learn the varying process noise covariance online. The A-KIT framework is applicable to any type of sensor fusion. Here, we present our approach to nonlinear sensor fusion based on an inertial navigation system and Doppler velocity log. By employing real recorded data from an autonomous underwater vehicle, we show that A-KIT outperforms the conventional EKF by more than 49.5\% and model-based adaptive EKF by an average of 35.4\% in terms of position accuracy.
## Dataset:
