function bounds = x2y2tohw(coords)
%X2Y2TOHW Does what a function with its name is probably meant to do (?)
bounds = [coords(1:2), coords(3:4) - coords(1:2)];
end

