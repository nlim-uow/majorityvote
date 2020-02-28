function [sign] = randSign(d)
sign=(double(rand(d,1)>0.5)*2)-1;
end