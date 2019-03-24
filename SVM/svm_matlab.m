close all
clear all

p=30; %h-index
q=29; %s-index
h=zeros(2,p);
s=zeros(2,q);
rng('default');
for i=1:2
dummyslump = rand(9,9);
end
for i=1:p
vinkel=(pi/2)*(1 + (i-1)/(p-1));
radie = 1+0.4*rand(1,1);
h(:,i) = [3 1]' + radie*[cos(vinkel) sin(vinkel)]';
end
for j=1:q
vinkel=(pi/2)*(3 + (j-1)/(q-1));
radie = 1+0.4*rand(1,1);
s(:,j) = [1 3]' + radie*[cos(vinkel) sin(vinkel)]';
end

figure(1)
clf
axis([1 3 1 3]);
plot(h(1,:),h(2,:),'bo')
hold on
plot(s(1,:),s(2,:),'ro')
hold on

options = optimset('GradObj','off','GradConstr','off',...
'Display','iter','Diagnostics','on');
fun = @(x) x(4)-x(3); %minimera istället för maximera
x0 = [0,0,0,0];
A = [[-h', ones(30,1), zeros(30,1)];...
    [s', zeros(29,1), -ones(29,1)]];
b = zeros(59,1);
Aeq = [];
beq = [];
lb = [0,-Inf,-Inf,-Inf];
ub = [Inf,0,Inf,Inf];
nonlcon = @begransning;

[wv,fval,exitflag] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon,options);

u1 = wv(1)
u2 = wv(2)
c = wv(3)
b = wv(4)

%plotta strecken
w1 = 2*u1/(c-b);
w2 = 2*u2/(c-b);
v = -(c+b)/(c-b);

rhs=0;
xx=[1 3]';
yy=[1 3]';
if w1+w2+v>rhs
yy(1)=(w1+v-rhs)/(-w2);
end
if w1+w2+v<rhs
xx(1)=(-w2-v+rhs)/w1;
end
if 3*w1+3*w2+v<rhs
yy(2)=(3*w1+v-rhs)/(-w2);
end
if 3*w1+3*w2+v>rhs
xx(2)=(-3*w2-v+rhs)/w1;
end

rhs=1;
xp=[1 3]';
yp=[1 3]';
if w1+w2+v>rhs
yp(1)=(w1+v-rhs)/(-w2);
end
if w1+w2+v<rhs
xp(1)=(-w2-v+rhs)/w1;
end
if 3*w1+3*w2+v<rhs
yp(2)=(3*w1+v-rhs)/(-w2);
end
if 3*w1+3*w2+v>rhs
xp(2)=(-3*w2-v+rhs)/w1;
end

rhs=-1;
xn=[1 3]';
yn=[1 3]';
if w1+w2+v>rhs
yn(1)=(w1+v-rhs)/(-w2);
end
if w1+w2+v<rhs
xn(1)=(-w2-v+rhs)/w1;
end
if 3*w1+3*w2+v<rhs
yn(2)=(3*w1+v-rhs)/(-w2);
end
if 3*w1+3*w2+v>rhs
xn(2)=(-3*w2-v+rhs)/w1;
end

plot(xp,yp,'b-')
hold on
plot(xx,yy,'g-')
hold on
plot(xn,yn,'r-')
legend('h', 's', 'L_+', 'L', 'L_-')
title('u2')
