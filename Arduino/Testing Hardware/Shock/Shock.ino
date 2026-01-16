
int shockPin = 5;  // SW-520D connected to D5

void setup() {
  pinMode(shockPin, INPUT_PULLUP); // Enable internal pull-up resistor
  Serial.begin(9600);
}

void loop() {
  int sensorState = digitalRead(shockPin);

  if (sensorState == LOW) { // Shock detected
    delay(5); // Debounce delay to avoid false triggers
    if (digitalRead(shockPin) == LOW) { // Confirm shock still present
      Serial.println("Shock Detected! Sending to PC...");
      delay(300); // Additional delay to prevent multiple rapid triggers
    }
  }
}