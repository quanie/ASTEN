mkdir bin
javac -d bin ./src/edu/uts/*.java
jar -cvf ASTEN.jar -C bin/ .
