// Define pins
const int dirPin = 13;   // Direction pin
const int pwmPin = 6;   // PWM pin

void setup() {
  // Initialize pins as output
  pinMode(dirPin, OUTPUT);
  pinMode(pwmPin, OUTPUT);
}

void loop() {
  // Set motor direction forward
  digitalWrite(dirPin, HIGH);
  
  // Set motor speed with PWM
  analogWrite(pwmPin, 128);  // Set PWM to 50% (128 out of 255)

}
