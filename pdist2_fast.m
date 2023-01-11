function D = pdist2_fast( X, Y, metric )
% Calculates the distance between sets of vectors.
% Piotr's Computer Vision Matlab Toolbox      Version 2.52
% Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
% Licensed under the Simplified BSD License [see external/bsd.txt]
%
% Revised by HD @ Feb 12, 2020.

if( nargin<3 || isempty(metric) ); metric=0; end;

switch metric
  case {0,'sqeuclidean'}
    D = distEucSq( X, Y );
  case 'euclidean'
    D = sqrt(distEucSq( X, Y ));
  case 'L1'
    D = distL1( X, Y );
  case 'cosine'
    D = distCosine( X, Y );
  case 'emd'
    D = distEmd( X, Y );
  case 'chisq'
    D = distChiSq( X, Y );
  otherwise
    error(['pdist2 - unknown metric: ' metric]);
end
D = max(0,D);
end

function D = distL1( X, Y )
m = size(X,1);  n = size(Y,1);
mOnes = ones(1,m); D = zeros(m,n);
for i=1:n
  yi = Y(i,:);  yi = yi( mOnes, : );
  D(:,i) = sum( abs( X-yi),2 );
end
end

function D = distCosine( X, Y )
% p=size(X,2);
% XX = sqrt(sum(X.*X,2)); 
% X = X ./ XX(:,ones(1,p));
X = bsxfun( @rdivide, X, sqrt(sum(X.*X,2)) );
% YY = sqrt(sum(Y.*Y,2)); 
% Y = Y ./ YY(:,ones(1,p));
Y = bsxfun( @rdivide, Y, sqrt(sum(Y.*Y,2)) );
D = 1 - X*Y';
end

function D = distEmd( X, Y )
Xcdf = cumsum(X,2);
Ycdf = cumsum(Y,2);
m = size(X,1);  n = size(Y,1);
mOnes = ones(1,m); D = zeros(m,n);
for i=1:n
  ycdf = Ycdf(i,:);
  ycdfRep = ycdf( mOnes, : );
  D(:,i) = sum(abs(Xcdf - ycdfRep),2);
end
end

function D = distChiSq( X, Y )
% note: supposedly it's possible to implement this without a loop!
m = size(X,1);  n = size(Y,1);
mOnes = ones(1,m); D = zeros(m,n);
for i=1:n
  yi = Y(i,:);  yiRep = yi( mOnes, : );
  s = yiRep + X;    d = yiRep - X;
  D(:,i) = sum( d.^2 ./ (s+eps), 2 );
end
D = D/2;
end

function D = distEucSq( X, Y )
Yt = Y';
% XX = sum(X.*X,2);
% YY = sum(Yt.*Yt,1);
% D = bsxfun(@plus,XX,YY)-2*X*Yt;
D = abs(bsxfun(@plus,sum(X.*X,2),sum(Yt.*Yt,1))-2*X*Yt);
end