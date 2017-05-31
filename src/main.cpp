#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// It's a duplicate value as it's present also in MPC.cpp...:(
const double Lf = 2.67;
// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

int main() {
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"];
          vector<double> ptsy = j[1]["ptsy"];
          double px = j[1]["x"];
          double py = j[1]["y"];
          double psi = j[1]["psi"];
          double v_mph = j[1]["speed"];
          // convert this value in meter per second
          double v = v_mph * (1609. / 3600.);

          double steer_value = j[1]["steering_angle"];
          double throttle_value = j[1]["throttle"];
          // Convert waypoints ptsx and ptsy in car coordinates...
          // |x'| = | cos(a) sin(a)  | * |x|
          // |y'| = | -sin(a) cos(a) | * |y|
          Eigen::VectorXd x(ptsx.size());
          Eigen::VectorXd y(ptsy.size());
          double sin_psi = sin(psi);
          double cos_psi = cos(psi);
          for (int i = 0; i < ptsx.size(); i++) {
            double mx = ptsx[i] - px;
            double my = ptsy[i] - py;
            x[i] = mx * cos_psi + my * sin_psi;
            y[i] = -mx * sin_psi + my * cos_psi;
          }

          // Define here the variable in milliseconds and in seconds
          int latency = 100;
          double latency_seconds = (double) latency / 1000;
          // After trasforming to car coord.. x0, y0 and psi are 0 so here
          // we take into account latency of the simulator.
          // Need to take in account orientation? of axis??
          psi = -v / Lf * steer_value * latency_seconds;
          double x0 =  v * cos(psi) * latency_seconds;
          double y0 = v * sin(psi) * latency_seconds;
          // not sure if I have to convert steer_value into grad here...or
          // radians is ok.
          v += throttle_value * latency_seconds;

          Eigen::VectorXd coeffs = polyfit(x, y, 3);
          double cte = polyeval(coeffs, x0) - y0; //that is 0;
          /// derivative of 3rd grade polynomio a*X^3 + b * X^2 + c* X + d -> c + 2 * b * x + 3 * a * X^2
          double epsi = psi -atan(coeffs[1] + 2*coeffs[2]* x0 + 3*coeffs[3]* x0* x0);

          Eigen::VectorXd state(6);
          // Take into account latency before optimizing values
          state(0) = x0;
          state(1) = y0;
          state(2) = psi;
          state(3) = v;
          state(4) = cte;
          state(5) = epsi;

          vector<double> vars = mpc.Solve(state, coeffs);

          steer_value = -vars[0]; // deg2rad(25);
          throttle_value = vars[1];

          json msgJson;
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = throttle_value;

          //Display the MPC predicted trajectory
          int limit = (vars.size() - 2)/ 2;
          vector<double> mpc_x_vals(limit);
          vector<double> mpc_y_vals(limit);

          for (int i = 0; i < limit; i++) {
            mpc_x_vals[i] = vars[i+2];
            mpc_y_vals[i] = vars[limit + 2 + i];
          }
          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Green line

          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;

          //Display the waypoints/reference line
          vector<double> next_x_vals;
          vector<double> next_y_vals;
          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Yellow line
          for(int i = 0 ; i < x.size(); i++) {
            next_x_vals.push_back(x(i));
            next_y_vals.push_back(y(i));
          }

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;


          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(latency));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
