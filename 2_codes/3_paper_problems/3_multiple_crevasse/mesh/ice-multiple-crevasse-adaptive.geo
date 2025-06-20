// Gmsh project created on Mon Nov 21 17:13:51 2022
SetFactory("OpenCASCADE");
lc = 20;
cw = 2.5;
Point(1) = {0, 0, 0, lc};
Point(2) = {500, 0, 0, lc};
Point(3) = {500, 125, 0, lc};
Point(4) = {0, 125, 0, lc};

delta_spacing = 50; // Spacing increase after each set

For i In {0:6} // Loop from 0 to 6, which makes 7 iterations
    c_spacing = (i+1) * delta_spacing;

    Point(6*i+5)  = {c_spacing, 125, 0, lc};
    Point(6*i+6)  = {c_spacing, 115, 0, lc};
    Point(6*i+7)  = {c_spacing + cw, 115, 0, lc};
    Point(6*i+8)  = {c_spacing + cw, 125, 0, lc};

    Point(6*i+9)  = {c_spacing + cw/2, 0, 0, lc};
    Point(6*i+10) = {c_spacing + cw/2, 125, 0, lc};
EndFor

Line(1) = {1, 2};
Line(3) = {3, 44};
Line(2) = {2, 3};
Line(4) = {44, 43};
Line(5) = {43, 42};
Line(6) = {42, 41};
Line(7) = {41, 38};
Line(8) = {38, 37};
Line(9) = {37, 36};
Line(10) = {36, 35};
Line(11) = {35, 32};
Line(12) = {32, 31};
Line(13) = {31, 30};
Line(14) = {30, 29};
Line(15) = {29, 26};
Line(16) = {26, 25};
Line(17) = {25, 24};
Line(18) = {24, 23};
Line(19) = {23, 20};
Line(20) = {20, 19};
Line(21) = {19, 18};
Line(22) = {18, 17};
Line(23) = {17, 14};
Line(24) = {14, 13};
Line(25) = {13, 12};
Line(26) = {12, 11};
Line(27) = {11, 8};
Line(28) = {8, 7};
Line(29) = {7, 6};
Line(30) = {6, 5};
Line(31) = {5, 4};
Line(32) = {4, 1};
Line(33) = {10, 9};
Line(34) = {16, 15};
Line(35) = {22, 21};
Line(36) = {28, 27};
Line(37) = {34, 33};
Line(38) = {40, 39};
Line(39) = {46, 45};

Curve Loop(1) = {32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
Plane Surface(1) = {1};


