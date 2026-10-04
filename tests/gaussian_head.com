%chk={checkpoint}.chk
%nprocshared=2
%mem=2GB
#sp b3lyp/6-311+G(3df) 

AnhDis curvilinear single-point calculation {job_tag}

0 2
