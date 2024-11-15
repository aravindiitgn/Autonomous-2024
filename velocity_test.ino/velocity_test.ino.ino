int enablePin = 10;   // PWM pin for speed control
int in1Pin = 2;      // Control pin for direction
int in2Pin = 3;      // Control pin for direction
int pwmValue = 0;    // Variable to store the current PWM value

void setup() {
  Serial.begin(9600);         // Initialize serial communication at 9600 bps
  pinMode(enablePin, OUTPUT);
  pinMode(in1Pin, OUTPUT);
  pinMode(in2Pin, OUTPUT);

  // Set motor direction (change as needed)
  digitalWrite(in1Pin, HIGH);
  digitalWrite(in2Pin, LOW);
  Serial.println("Enter a PWM value between 0 and 255:");
}

void loop() {
  if (Serial.available() > 0) {           // Check if there's input from the Serial Monitor
    int newValue = Serial.parseInt();     // Read the entered PWM value
    newValue = constrain(newValue, 0, 255); // Ensure PWM value is within 0-255 range

    if (newValue != pwmValue) {           // Only update if the new PWM value is different
      pwmValue = newValue;
      analogWrite(enablePin, pwmValue);   // Set PWM output based on entered value
      Serial.print("PWM set to: ");
      Serial.println(pwmValue);           // Print the set PWM value for confirmation
    }
    // Clear the Serial input buffer
    while (Serial.available() > 0) {
      Serial.read();
    }
  }
}
