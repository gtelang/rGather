clc;
clear;

% Extract data from the file
data = load('finalShenzhen9386V6.mat');
lats  = data.lat  ; 
longs = data.long ;  

samplesPerCar = size(longs,1) ;
numCars       = size(longs,2) ;


% Plot the data of some cars 
figure(1)

endTime           = 3;
numCarsConsidered = 100;

% Axis limits
xmin = min(lats(:))  ;
xmax = max(lats(:)) ;

ymin = min(longs(:)) ;
ymax = max(longs(:)) ;

for car = 1 : numCarsConsidered
    
    randomColor = [rand,rand,rand]; % Stick to RGB for now. Change to HSV for better rendering later. 
    
  
    xs = lats (1:endTime,car);
    ys = longs(1:endTime,car);
    
    p = plot( xs, ys  ,'s-','linewidth',4,'markerfacecolor',randomColor);
    p.Color(4) = 0.2;
    
    % Set axis to a fixed scale every-single time
    axis ([xmin, xmax, ymin, ymax])
    
    hold on
end
hold off