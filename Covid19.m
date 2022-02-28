%% Use Regression to analyze vaccine coverage
%% Read data
data = readtable('dataEng.csv');
%% Pick out X and Y
Y = data.Vaccinated;
%Pick out explanatory variables in X
X = table2array(data(:,3:end));
x = table2array(data(:,3:end)); %Without ones on the first Column, we use this in stepwise
%Add an intercept to X
X = [ones(size(X,1),1) X];
%Names on variables
variables = data.Properties.VariableNames(3:end);

%% Find Lund (Vaccinated=NaN)
I = isnan(Y);
%Pick out Lund from X Matrix
X_Lund = X(I,:);
%And keep other X (~I -> Not Lund)
X = X(~I,:);
x = x(~I,:); 
%Same for Y
Y_Lund = Y(I,:);
Y = Y(~I,:);

%% Examine size of Variables
whos X Y X_Lund Y_Lund variabler
%We should have 290-1=289 muncipalities (ie rows) in X and Y, and 16 columns in X
%with our 15 variables + one column with zeros (intercept).

%% Plot data as function of variables 
%index i+1 in X because we don't want the first column(the intercept).
figure(1)
for i=1:length(variables)
  subplot(3,5,i)
  plot(X(:,i+1),Y,'.')
  title( variables{i} )
end

%Alternativly transform probabilities in Y (must be in range 0-1) to
%real numbers with a logit transform
logitY = log(Y)-log(1-Y);
%We get probabilities as Y=1/(1+exp(-logitY))
%Plot the relationship in logit scale.
figure(2)
for i=1:length(variables)
  subplot(3,5,i)
  plot(X(:,i+1), logitY,'.')
  title( variables{i} )
end
%% Model
stepwise(x, logitY); %Here we find which variables are intereseting, and which are not
% Remove uninteresting from X.
X(:,14) = [];
X(:,12) = [];
X(:,11) = [];
X(:,9) = [];
X(:,8) = [];
X(:,7) = [];
X(:,2) = [];

%Also remove from X_lund.
X_Lund(14) = [];
X_Lund(12) = [];
X_Lund(11) = [];
X_Lund(9) = [];
X_Lund(8) = [];
X_Lund(7) = [];
X_Lund(2) = [];

%Also remove from variables. (remeber that we don't have an intercept here)
variables(13) = [];
variables(11) = [];
variables(10) = [];
variables(8) = [];
variables(7) = [];
variables(6) = [];
variables(1) = [];
%% Determine the regression
beta = X\logitY; 
res = logitY-X*beta; %residuals
[n, c] = size(X); %nbr of rows and columns
f = n-c;
s2 = sum(res.^2)/f; %estimation of sigma^2

%Plot residuals as they come and in a normplot
figure(3);
plot(res, '*')
ylabel('e')
xlabel('1:n')
title('Resiudals')
figure(4);
normplot(res)


logitY_est = X_Lund * beta; %Estimated log transformed vaccinsation coverage
g = @ (x) 1./(1+exp(-x));
Y_est = g(logitY_est) %Transform back to probability

V_logitY =  s2 .*(1+ X_Lund*inv(X'*X)*X_Lund');
d_logitY = sqrt(V_logitY);
I = logitY_est +[-1 1].*tinv(0.975, f) .* d_logitY; %prediction interval for log transformed vaccination coverage.
I = arrayfun(g , I) %transform back to probability

%% Plot residuals agiant choosen explanatory variables and as they come
figure(5)
subplot(3,3,1);
    plot(res, '*');
    t2=title('Residuals "as they come"');
    set(t2, 'FontSize', 7);
for k = 1:8
    subplot(3,3,k+1);
    plot(X(:,k+1), res, '*');
    t= title('Residuals against ' + string(variables(k)));
    set(t, 'FontSize', 7);
end

%% Analysis of risk reduction
N = 10.^5;
elogit = normrnd(3.18, 0.73, N, 1); %Simulated transformed efficacy
e = arrayfun(g, elogit); %transform back
P_vlogit = normrnd(logitY_est, sqrt(V_logitY), N, 1); %simulated transformed vaccination coverages
P_v = arrayfun(g, P_vlogit); %transform back

RR_i = times((1-e),P_v) + (1-P_v); 
RR_est = mean(RR_i) %Mean of RR

upperquantile = quantile(RR_i, 0.975);
lowerquantile = quantile(RR_i, 0.025);
I_RR=[lowerquantile upperquantile] % confidence intervall
%% histogram plot of RR
bin_edges = 0:0.002:0.8;
figure(6)
normplot(RR_i) %We see that RR is not normal dsitributed!
figure(7)
histogram(RR_i(RR_i<lowerquantile),bin_edges,'FaceColor','r') %Part of histogram under the lower quantile
hold on
histogram(RR_i(RR_i<=upperquantile & RR_i>=lowerquantile),bin_edges, 'FaceColor','b') %Part in the confidence interval
hold on
histogram(RR_i(RR_i>upperquantile),bin_edges,'FaceColor','r') %Part of histogram over the upper quantile

title('Histogram over simulated risk reductions')
xlabel('Risk Reduction')
ylabel('Nbr')