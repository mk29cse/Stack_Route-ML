This repository contains implementation of Linear Regression using scikit- linearmodel and statsmodels- ols model.
##### Data Set 01: Planetary Motion Data #####
 

##### Description: #####



Researchers in a University conducted a series of experiments for predicting the orbital period of Earth and other planets. Different measures such as semi-major axis, volume, density etc.., along with the orbital period of the planets were obtained.

Problem:

Predict the orbital period of the planets from the data and check if they confirm to Kepler’s Third Law of Planetary Motion.





##### Generative Model: #####
 

$$
Orbital\,Period(in\,years)=1(Semi\,Major\,Axis)^{1.5}
$$

#####  Data contains: #####
 

Target : Orbital period of the planet (in years)

Feature 1: Semi Major Axis (in AU)

Feature 2: Volume

Feature 3: Escape Velocity

Feature 4 : Mean Density

##### Data Set 02: Tree Age Data #####
 

##### Description:#####
 



Maple is the most commonly used wood for making Violin.The sound quality depends upon the age of the wood used. A group of Violin makers collected sample of Maple trees and tried to predict the age of the wood using ‘Radio carbon dating’ process. Carbon 14 isotop was used in the carbon dating process.

Regression Problem: Predict the age of the wood

Predict the age of wood sample using the given measures.

Classification problem: Classify Maple tree as ‘Good’ quality or ‘Bad’ quality

We have to classify Maple tree as “Good”" quality wood or “Bad” quality wood for making a Violin. If the age of the tree is greater than 250 years, then it is considered as ‘Good’ quality.



##### Generative Model: #####



$$
Age\, of\, wood  = \frac{2.303}{Decay\,constant}\log\frac{14C\,activity\,in\,a\,fresh\,sample}{14C\,activity\,at\,the\,age\, t}
$$

##### Data contains: ##### 

Target : Age of wood in Regression problem (in years)

Feature 1: N0 (14 C activity in a fresh sample)

Feature 2: Nt (14 C activity at the age t)

Feature 3: pH value of the soil

Feature 4: Number of Sepals in Maple flower

##### Data Set 03: Systolic Blood Pressure Data #####
 

##### Description: ######
 



Lab-techinicians in a laboratory decided to assess the association between BMI and Systolic blood pressure. Additionally, they collected data on Age, Body weight,and treatment for hypertension of 100 persons.

Regression Problem: Predict systolic blood pressure

Predict blood pressure based on BMI, BodyWeight, Age , Treatment for hypertension, and other covariates measured for that person.

Classification Problem: Predict whether person has high blood pressure or not

Normal systolic blood pressure is 90-120 mm Hg. A BP greater than 120 mm Hg is considered as pre-high blood pressure. Classify a persons as having ‘Normal’ or ‘Abnormal’ BP.




##### Generative Model: #####
 

$$
SystolicBP = 60+0.58\,BMI+0.65\,Age+6.44(Treatment\,for\,hypertension)+0.7(Height\,of\,a\,person)
$$

##### Data contains : #####

Target: Systolic blood pressure( in mm Hg)

Feature 1: Age

Feature 2: Treatment for hypertension (1-Yes,0-No)

Feature 3: Gender (1-Yes,0-No)

Feature 4: Number of members in the family

Feature 5: Food habit (1-Vegan,2-Any)

Feature 6: Weight of a person (in Kg)

Feature 7: Height of a person (in m)

Feature 8: ScaledBMI



##### Data Set 04: mpg Data #####
 

##### Description: ####
 



Data describes different car models by their characteristics such as displacement,weight etc and their mileage is also recorded. We like to see the association between fuel efficiency (measured in miles per gallon) and the car model and make and other attributes.

Prediciton problem :

Given a car’s make, model and other attributes, predict the fuel efficiency.

Classification problem : We have to classify the car as ‘Desirable quality’ or ‘Undesirable quality’ based on mpg

We have to classify a car as a Gas guzzler or not. If ‘mpg’ is less than average mpg, then it is considered as “GasGuzzler” otherwise not (FuelEfficient).



##### Data contains:#####



Target: mpg

Feature 1: Cylinders

Feature 2: Displacement

Feature 3: Horsepower

Feature 4: Weight of the driver (classification problem) ; weight of the car(Regression problem)

Feature 5: Acceleration

Feature 6: Model year

Feature 7: Origin

Feature 8: Car name


##### Data Set 05: Ionosphere Data #####
 

##### Description: #####




This radar data was collected by a system in Goose Bay, Labrador. The system consists of a phased array of multiple antennas that sends out signal every 5s all round the year. Each signal here is represented using 17 pulse numbers for the Goose Bay system with 2 attributes per pulse number.The aim of experiment is to detect free electrons in the Ionosphere.

Classification problem

We have to classify the readings from the radar as “Good” or “Bad”.“Good” readings will returns are those showing evidence of some type of structure in the ionosphere. “Bad” readings does not show any evidence of structure.

##### Data Set 06: Pizza Hut Data #####


##### Description: #####




We have the sales data of pizza in Central Bengaluru for the last 100 months. Usual per month sales in this area is 16000-17500 pizza per month.

Regression problem: Prediction of pizza demanded

Predict the demand for Pizza in Central Bengaluru based on Pizza price, Burger price and other variables.

Classification problem: Classification of month as ‘weak’ or ‘Good’ based on the sales

By analysing the sales data in the last 100 months, we want to classify the months as ‘weak’ or ‘Good’. If the sales is less than 15500( less than minimum of usual sales) , it is considerd as ‘weak’ month in sales, otherwise considered as ‘Good’ month.

##### Generative Model: #####


$$
y=15000−5(Price\,of\,pizza)+14(Price\,of\,burger)−15(Price\,of\,soft\,drinks)+28(Income\,of\,the\,person)
$$

##### Data contains: ##### 

Target: Number of pizza demanded.

Feature 1: Price of pizza

Feature 2: Price of burger

Faeture 3: Room temperature

Feature 4: Delivery mode(1-Home,2-Shop)

Feature 5: Income of the person(in 1000)

Feature 6: Price of softdrinks

Feature 7: Location (1- With-in-City-Limits, 2- Outskirts)

##### Data Set 07: Sonar Data #####
 

##### Description: #####




Scientists have divided the ocean into five main layers. These layers, known as “zones”, extend from the surface to the most extreme depths where light can no longer penetrate. A submarine emits sonar pulses to different parts of ocean, which returns from an underwater cliff/object.The speed of ultrasonic waves in sea-water is 1250 m/s. Based on the distance to the object, we have to classify the objects as in a particular zone or not.

Regression problem: Calculate the distance to the object from the ocean surface

Based on the sonar pulses echo receiving time and other covariate variables, calculate the distance to the object from the ocean level.

Classification problem: Classify the objects based on their location

The fourth layer of the ocean is known as the ‘Abyssopelagic zone’ and extends from the 4000-6000 meters. Classify objects as in ‘Abyssopelagic Zone’or not. If the distance to the object from the surface is between 4000-6000 meters, then we can say that ’object’ is in ‘Abyssopelagic’ zone.




##### Generative Model: #####
 

$$
y=\frac{1250(Echo\,receiving\,time)}{2}
$$

##### Data contains: #####
 


Target : Depth from the surface (in meter)

Feature 1: Echo receiving time (in seconds)

Feature 2: Water temperature (in degree celsius)

Feature 3: concentration of cl-ion (usually 19.2632)

Feature 4: Salinity (salinity=1.80655*Chlorinity)

Feature 5: pH of sea water (Usually 7.5-8.4)

##### Data Set 08: Iris Data #####


##### Description: #####
 



The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimetres. Based on the combination of these four features, Fisher developed a linear discriminant model to distinguish the species from each other.

Classification problem

Based on the characteristics such as Sepal length,Sepal width,Petal length and Petal width, we have to classify Iris plant into ‘Setosa’,‘Versicolour’,‘Virginica’.



###### Data contains: ######
 


Feature 1: Sepal length (in cm )

Feature 2: Sepal width (in cm )

Feature 3: Petal length (in cm)

Feature 4: Petal width (in cm )
