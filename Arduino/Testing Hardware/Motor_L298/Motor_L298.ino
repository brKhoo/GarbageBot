
/* // Defining the motor control pins here
###define IN1 6  
#define IN2 7  
#define IN3 8  
#define IN4 9  

void setup() {
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
}

void loop() {
  // Moves forward now
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  delay(2000); // runs for 2 s

  // stops the motors now
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  delay(1000); // pauses for 1s

  // moves backward now
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
  delay(2000); // runs for 2s

  // stops the motors now
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  delay(1000); // pauses for 1s
}*/


// Defining the motor control pins
#include <Arduino.h> // Include the Arduino library

// Defining the motor control pins
const int int1 = 6; // Motor 1 pin
const int int2 = 7; // Motor 2 pin
const int int3 = 8; // Motor 3 pin 
const int int4 = 9; // Motor 4 pin 

// Ultrasonic sensor pins
const int trigPin = 10; // Trigger pin
const int echoPin = 11; // Echo pin

long duration; // Variable that stores the duration of the echo
int distance;  // Variable that stores the calculated distance

void setup() {
  
  pinMode(int1, OUTPUT); // Sets the motor control pins as outputs now
  pinMode(int2, OUTPUT);
  pinMode(int3, OUTPUT); // Setting int3 as output
  pinMode(int4, OUTPUT); // Setting int4 as output
  
  // Set ultrasonic sensor pins
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  
  // Start serial communication
  Serial.begin(115200); // for faster communication
}

void loop() {
  // Triggering the ultrasonic sensor
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  // Measures the echo duration
  duration = pulseIn(echoPin, HIGH);
  distance = duration * (0.034 / 2); // Calculate distance in cm

  // Prints the distance for debugging
  Serial.print("Distance: ");
  Serial.println(distance);

  // Controls the motors based on distance (I used if statements)
  if (distance < 20) {
    // Stop all motors if the object is too close
    digitalWrite(int1, LOW);
    digitalWrite(int2, LOW);
    digitalWrite(int3, LOW);
    digitalWrite(int4, LOW);
  } 
  else if (distance >= 20 && distance < 50) {
    // Move forward slowly using motor 1 and motor 2
    digitalWrite(int1, HIGH); // Activate motor 1
    digitalWrite(int2, LOW);  // Deactivate motor 2
    digitalWrite(int3, LOW);  // Deactivate motor 3
    digitalWrite(int4, LOW);  // Deactivate motor 4
  } 
  else {
    // Move forward fast using motor 1 and motor 3
    digitalWrite(int1, HIGH); // Activate motor 1
    digitalWrite(int2, LOW);  // Deactivate motor 2
    digitalWrite(int3, HIGH); // Activate motor 3
    digitalWrite(int4, LOW);  // Deactivate motor 4
  }

  delay(500); // Delay for stability
}

