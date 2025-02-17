// ALL SENSORS COMBINED!

//#include <SoftwareSerial.h>
#include <Servo.h>

//SoftwareSerial BT(2, 3); // HC-05 TX to D2, RX to D3

Servo myservo;  // create servo object to control a servo

const int int1 = 6; // Motor 1 pin
const int int2 = 7; // Motor 2 pin
const int int3 = 8; // Motor 3 pin 
const int int4 = 9; // Motor 4 pin 

const int shockPin = 5;  // SW-520D connected to D5

// Ultrasonic sensor pins
const int trigPin = 10; // Trigger pin
const int echoPin = 11; // Echo pin

int requiredDistance = 0;
long duration; // Variable that stores the duration of the echo
int distance;  // Variable that stores the calculated distance
int range = 5;
bool moving = false;

void writeMotor(int speed) {
  if(speed == 0) {
    digitalWrite(int1, LOW);
    digitalWrite(int2, LOW);  
    digitalWrite(int3, LOW);
    digitalWrite(int4, LOW);
  } else if(speed > 0) {
    analogWrite(int1, speed);
    analogWrite(int2, 0);
    analogWrite(int3, speed);
    analogWrite(int4, 0);
  } else {
    digitalWrite(int1, 0);
    digitalWrite(int2, -speed);
    digitalWrite(int3, 0);
    digitalWrite(int4, -speed);
  }
}

void setup() {
  // setup motors!
  pinMode(int1, OUTPUT); // Sets the motor control pins as outputs now
  pinMode(int2, OUTPUT);
  pinMode(int3, OUTPUT); // Setting int3 as output
  pinMode(int4, OUTPUT); // Setting int4 as output

  // setup vibration!
  pinMode(shockPin, INPUT_PULLUP); // Enable internal pull-up resistor

  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  myservo.attach(4); 

//  // setup bluetooth!
//  BT.begin(9600);  // Start Bluetooth communication

  // setup serial communication
  Serial.begin(9600);
}

void loop() {

  // check bluetooth
//  if(BT.available()) {
//    String data = BT.readString();  

  if(Serial.available()) {
    String data = Serial.readString();  


    data.trim();

//    Serial.println("Recieved data from PC...");
//    Serial.print(" -> Got data ");
//    Serial.println(data);

    
    moving = true;

    if(data == String("LOCATION1")) {
      requiredDistance = 10;  
    } else if(data == String("LOCATION2")) {
      requiredDistance = 20;
    } else if(data == String("LOCATION3")) {
      requiredDistance = 30;  
    } else {
      moving = false;  
    }

//    Serial.print("Target Distance: ");
//    Serial.println(requiredDistance);
//    Serial.print("Current Distance: ");
//    Serial.println(distance);

  }


  // check position
  digitalWrite(trigPin,LOW);
  delayMicroseconds(2);

  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin,LOW);
  
  // Measures the echo duration
  duration = pulseIn(echoPin, HIGH);
  distance = duration * (0.034 / 2); // Calculate distance in cm

  if(distance != 0 && moving == true) {
//    Serial.println(distance);
    if(distance > requiredDistance + range) {
      writeMotor(-255);
    } else if(distance < requiredDistance - range) {
      writeMotor(255);
    } else {
      writeMotor(0);

      // dump garbage
//      Serial.println("DUMP");
      myservo.writeMicroseconds(1500);
      delay(500);
      myservo.writeMicroseconds(1700);
      delay(500);
      myservo.writeMicroseconds(1500);
      delay(1000);
      myservo.writeMicroseconds(1300);
      delay(500);
      myservo.writeMicroseconds(1500);

      moving = false;.
    }
  }

  int sensorState = digitalRead(shockPin);

  if (sensorState == LOW) { // Shock detected
    delay(5); // Debounce delay to avoid false triggers
    if (digitalRead(shockPin) == LOW) { // Confirm shock still present
//      Serial.println("Shock Detected! Sending to PC...");
//      BT.println("Shock Detected"); // Send message via Bluetooth
      Serial.println("Shock Detected");
      delay(300); // Additional delay to prevent multiple rapid triggers
    }
  }
}// ALL SENSORS COMBINED!

//#include <SoftwareSerial.h>
#include <Servo.h>

//SoftwareSerial BT(2, 3); // HC-05 TX to D2, RX to D3

Servo myservo;  // create servo object to control a servo

const int int1 = 6; // Motor 1 pin
const int int2 = 7; // Motor 2 pin
const int int3 = 8; // Motor 3 pin 
const int int4 = 9; // Motor 4 pin 

const int shockPin = 5;  // SW-520D connected to D5

// Ultrasonic sensor pins
const int trigPin = 10; // Trigger pin
const int echoPin = 11; // Echo pin

int requiredDistance = 0;
long duration; // Variable that stores the duration of the echo
int distance;  // Variable that stores the calculated distance
int range = 5;
bool moving = false;

void writeMotor(int speed) {
  if(speed == 0) {
    digitalWrite(int1, LOW);
    digitalWrite(int2, LOW);  
    digitalWrite(int3, LOW);
    digitalWrite(int4, LOW);
  } else if(speed > 0) {
    analogWrite(int1, speed);
    analogWrite(int2, 0);
    analogWrite(int3, speed);
    analogWrite(int4, 0);
  } else {
    digitalWrite(int1, 0);
    digitalWrite(int2, -speed);
    digitalWrite(int3, 0);
    digitalWrite(int4, -speed);
  }
}

void setup() {
  // setup motors!
  pinMode(int1, OUTPUT); // Sets the motor control pins as outputs now
  pinMode(int2, OUTPUT);
  pinMode(int3, OUTPUT); // Setting int3 as output
  pinMode(int4, OUTPUT); // Setting int4 as output

  // setup vibration!
  pinMode(shockPin, INPUT_PULLUP); // Enable internal pull-up resistor

  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  myservo.attach(4); 

//  // setup bluetooth!
//  BT.begin(9600);  // Start Bluetooth communication

  // setup serial communication
  Serial.begin(9600);
}

void loop() {

  // check bluetooth
//  if(BT.available()) {
//    String data = BT.readString();  

  if(Serial.available()) {
    String data = Serial.readString();  


    data.trim();

//    Serial.println("Recieved data from PC...");
//    Serial.print(" -> Got data ");
//    Serial.println(data);

    
    moving = true;

    if(data == String("LOCATION1")) {
      requiredDistance = 10;  
    } else if(data == String("LOCATION2")) {
      requiredDistance = 20;
    } else if(data == String("LOCATION3")) {
      requiredDistance = 30;  
    } else {
      moving = false;  
    }

//    Serial.print("Target Distance: ");
//    Serial.println(requiredDistance);
//    Serial.print("Current Distance: ");
//    Serial.println(distance);

  }


  // check position
  digitalWrite(trigPin,LOW);
  delayMicroseconds(2);

  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin,LOW);
  
  // Measures the echo duration
  duration = pulseIn(echoPin, HIGH);
  distance = duration * (0.034 / 2); // Calculate distance in cm

  if(distance != 0 && moving == true) {
//    Serial.println(distance);
    if(distance > requiredDistance + range) {
      writeMotor(-255);
    } else if(distance < requiredDistance - range) {
      writeMotor(255);
    } else {
      writeMotor(0);

      // dump garbage
//      Serial.println("DUMP");
      myservo.writeMicroseconds(1500);
      delay(500);
      myservo.writeMicroseconds(1700);
      delay(500);
      myservo.writeMicroseconds(1500);
      delay(1000);
      myservo.writeMicroseconds(1300);
      delay(500);
      myservo.writeMicroseconds(1500);

      moving = false;.
    }
  }

  int sensorState = digitalRead(shockPin);

  if (sensorState == LOW) { // Shock detected
    delay(5); // Debounce delay to avoid false triggers
    if (digitalRead(shockPin) == LOW) { // Confirm shock still present
//      Serial.println("Shock Detected! Sending to PC...");
//      BT.println("Shock Detected"); // Send message via Bluetooth
      Serial.println("Shock Detected");
      delay(300); // Additional delay to prevent multiple rapid triggers
    }
  }
}