/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;  // NOTE_AV: added new
using std::uniform_int_distribution;  // NOTE_AV: added new
using std::uniform_real_distribution;  // NOTE_AV: added new


// Random number generator
static std::default_random_engine gen;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  num_particles = 100;  // Set the number of particles.

  // Create a normal (Gaussian) distribution for x, y, and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    particles.push_back(p);
  }
  is_initialized = true;
}


void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *   http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *   http://www.cplusplus.com/reference/random/default_random_engine/
   */
  double x, y, theta;

  for (int i = 0; i < num_particles; ++i) {

    if (fabs(yaw_rate) > 1e-5) {
      // Turning, angular velocity is measurable.
      theta = particles[i].theta + (yaw_rate * delta_t);
      x = particles[i].x + (velocity / yaw_rate) * (sin(theta) - sin(particles[i].theta));
      y = particles[i].y + (velocity / yaw_rate) * (cos(particles[i].theta) - cos(theta));
    } else {
      // Driving in a straight line.
      theta = particles[i].theta;
      x = particles[i].x + (velocity * delta_t) * cos(theta);
      y = particles[i].y + (velocity * delta_t) * sin(theta);
    }

    // Create a normal (Gaussian) distribution for x, y, and theta
    normal_distribution<double> dist_x(x, std_pos[0]);
    normal_distribution<double> dist_y(y, std_pos[1]);
    normal_distribution<double> dist_theta(theta, std_pos[2]);

    particles[i].x = dist_x(gen);  // NOTE_AV: should it be 'x + dist_x(gen)'?
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */
  double rmse;
  double min_rmse = std::numeric_limits<double>::max();
  int num_obs = predicted.size();

  for (int i = 0; i < num_obs; ++i) {
    rmse = dist(predicted[i].x, predicted[i].y, observations[i].x, observations[i].y);
    if (min_rmse > rmse) {
        observations[i].id = predicted[i].id;  // Update an observation's id with nearest landmark's id.
        min_rmse = rmse;
    }
  }
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   *
   * Update Weights::
   *   Step 1: Transformation
   *   Step 2: Association
   *   Step 3: Update Weights
   */

  // Used in multivariate normal (Gaussian) distribution PDF (probability density function) calculation.
  double std_x = std_landmark[0];
  double std_y = std_landmark[1];
  double divisor = 2 * M_PI * std_x * std_y;
  double std_x2 = 2 * std_x * std_x;
  double std_y2 = 2 * std_y * std_y;

  double norm = 0.0;

  for (int i = 0; i < num_particles; ++i) {  // START iteration over particles.
    double x_p, y_p, theta;
    vector<LandmarkObs> landmarks_within_range;

    x_p = particles[i].x;
    y_p = particles[i].y;
    theta = particles[i].theta;

    // Find map landmarks within car's sensor range.
    for (int j = 0; j < map_landmarks.landmark_list.size(); ++j) {  // START iteration over map_landmarks for each particle.
      double x_ml, y_ml, rmse;
      int id_ml;

      id_ml = map_landmarks.landmark_list[j].id_i;
      x_ml = map_landmarks.landmark_list[j].x_f;
      y_ml = map_landmarks.landmark_list[j].y_f;
      rmse = dist(x_p, y_p, x_ml, y_ml);
      if (rmse <= sensor_range) {
        landmarks_within_range.push_back(LandmarkObs {id_ml, x_ml, y_ml});
      }
    }  // END map_landmarks loop.

    // Step 1: Transformation
    vector<LandmarkObs> observations_map;
    for (int j = 0; j < observations.size(); ++j) {  // START iteration over observations.
      double x_o, y_o, x_om, y_om;
      int id_o;

      id_o = observations[j].id;
      x_o = observations[j].x;
      y_o = observations[j].y;
      x_om = x_p + (x_o * cos(theta)) - (y_o * sin(theta));
      y_om = y_p + (x_o * sin(theta)) + (y_o * cos(theta));
      observations_map.push_back(LandmarkObs {id_o, x_om, y_om});
    }  // END observations loop.

    // Step 2: Association
    // Associate observations in map coordinates (i.e. Transformed observations) to
    // map landmarks within car's sensor range for each particle.
    dataAssociation(landmarks_within_range, observations_map);

    // Step 3: Update Weights
    double likelihood = 1.0;

    for (int j = 0; j < observations_map.size(); ++j) {  // START iteration over observations in map coordinates.
      double x_om, y_om, x_ml, y_ml;
      int id_om;

      id_om = observations_map[j].id;
      x_om = observations_map[j].x;
      y_om = observations_map[j].y;
      x_ml = 0.0;
      y_ml = 0.0;

      for (int k = 0; k < landmarks_within_range.size(); ++k) {  // START iteration over map landmarks within car's sensor range.
        if (id_om == landmarks_within_range[k].id) {
          x_ml = landmarks_within_range[k].x;
          y_ml = landmarks_within_range[k].y;
          break;
        }
      }  // END landmarks_within_range loop.

      // Multivariate normal (Gaussian) distribution PDF (probability density function) calculation.
      double exponent = pow(x_om - x_ml, 2) / std_x2 + pow(y_om - y_ml, 2) / std_y2;
      likelihood *= exp(-exponent) / divisor;
    }  // END observations_map loop.

    particles[i].weight = likelihood;
    norm += likelihood;  // Same as particles[i].weight
  }  // END num_particles loop.

  // Normalize particles' weight
  norm += std::numeric_limits<double>::epsilon();
  for (int j = 0; j < num_particles; ++j) {
    particles[j].weight /= norm;
  }
}


void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<double> weights;
  double max_weight = std::numeric_limits<double>::min();

  for (int j = 0; j < num_particles; ++j) {
    double weight = particles[j].weight;
    weights.push_back(weight);
    if (weight > max_weight) {
      max_weight = weight;
    }
  }

  uniform_int_distribution<int> dist_index(0, num_particles - 1);
  uniform_real_distribution<double> dist_weights(0.0, max_weight);

  double beta = 0.0;  // Weight representation circle

  int w_i = dist_index(gen);
  vector<Particle> resample_particles;

  for (int j = 0; j < num_particles; ++j) {
    beta += 2.0 * dist_weights(gen);
    while (beta > weights[w_i]) {
      beta -= weights[w_i];
      w_i = (w_i + 1) % num_particles;
    }
    resample_particles.push_back(particles[w_i]);
  }

  particles = resample_particles;

  // NOTE_AV: reset particle weights?
  //for (int j = 0; j < num_particles; ++j) {
  //  particles[j].weight = 1.0;
  //}
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
