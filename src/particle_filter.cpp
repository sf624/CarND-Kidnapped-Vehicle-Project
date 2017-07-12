/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 100;
	is_initialized = true;

	// Create gaussian distribution for x, y and theta.
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]), 
															dist_y(y, std[1]), 
															dist_theta(theta, std[2]);


	for (int i = 0; i < num_particles; ++i) {
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1;

		particles.push_back(particle);
		weights.push_back(particle.weight);
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// Use bicycle model for prediction
	for (int i = 0; i < num_particles; ++i) {
		if (abs(yaw_rate) <= 0.0001) {
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
			// particles[i].theta += yaw_rate * delta_t;
		}
		else {
			particles[i].x += velocity / yaw_rate*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}

		// Add gaussian noise
		default_random_engine gen;
		normal_distribution<double> dist_x(particles[i].x, std_pos[0]*0.001),
																dist_y(particles[i].y, std_pos[1]*0.001),
																dist_theta(particles[i].theta, std_pos[2]*0.001);

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int i=0; i< observations.size(); ++i) {
		double min_dist2 = 1e6;
		double o_x = observations[i].x;
		double o_y = observations[i].y;
		for (int j = 0; j < predicted.size(); ++j) {
			double p_x = predicted[j].x;
			double p_y = predicted[j].y;
			double dist2 = (o_x - p_x) * (o_x - p_x) + (o_y - p_y) * (o_y - p_y);

			if (dist2 < min_dist2) {
				observations[i].id = predicted[j].id;
				min_dist2 = dist2;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
	std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (int i=0; i<num_particles; ++i) {
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;

		std::vector<LandmarkObs> predicted;
		
		// Calculate map_landmarks in vehicle's cooridnate assuming particle's state.
		for (int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
			double m_x = map_landmarks.landmark_list[j].x_f;
			double m_y = map_landmarks.landmark_list[j].y_f;

			LandmarkObs landmark;
			landmark.x = cos(-theta) * (m_x - x) - sin(-theta) * (m_y - y);
			landmark.y = sin(-theta) * (m_x - x) + sin(-theta) * (m_y - y);
			landmark.id = map_landmarks.landmark_list[j].id_i;
			predicted.push_back(landmark);
		}

		// Associate observation with map_landmark (estimated in vehicle coordinate).
		dataAssociation(predicted, observations);

		// Calculate weight using multi-variate Gaussian probability.
		particles[i].weight = 1;
		for (int j = 0; j < observations.size(); ++j) {
			double sig_x2 = std_landmark[0]*std_landmark[0];
			double sig_y2 = std_landmark[1]*std_landmark[1];
			double dx = 0, dy = 0;

			// Search associated landmark and calculate difference.
			for (int k = 0; k < predicted.size(); ++k) {
				if (observations[j].id == predicted[k].id) {
					double dx = observations[j].x - predicted[k].x;
					double dy = observations[j].y - predicted[k].y;
					break;
				}
			}

			particles[i].weight *= exp(-0.5 * (1.0/sig_x2*dx*dx + 1.0/sig_y2*dy*dy))
														/ sqrt(2.0*M_PI*sig_x2*sig_y2);
		}
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	for (int i = 0; i < weights.size(); ++i) {
		weights[i] = particles[i].weight;
	}

	random_device seed_gen;
	mt19937 engine(seed_gen());

	discrete_distribution<int> dist(weights.begin(), weights.end());

	std::vector<Particle> old_particles = particles;
	for (int i = 0; i < particles.size(); ++i) {
		particles[i] = old_particles[dist(engine)];
	}
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	
	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
