module ansiportlist(
input [3:0] in_1,input [3:0] in_2, output [3:0]	or_o, output [3:0] nor_o, output [3:0] and_o, output [3:0] nand_o, output [3:0] xor_o,  output [3:0] xnor_o,output [3:0] add_o, output [3:0] sub_o,output [3:0] mul_o, output [3:0] equ_o, output [3:0] neq_o, output [3:0] gt_o, output [3:0] lt_o,output [3:0] geq_o, output [3:0]leq_o
); 

assign or_o    = in_1  | in_2; 
assign nor_o   = in_1 ~| in_2; 
assign and_o   = in_1  & in_2; 
assign nand_o  = in_1 ~& in_2;  
assign xor_o   = in_1  ^ in_2;
assign xnor_o  = in_1 ~^ in_2;  

assign add_o   = in_1  + in_2; 
assign sub_o   = in_1  - in_2;

assign mul_o   = in_1  * in_2;  

assign equ_o   = in_1 == in_2; 
assign neq_o   = in_1 != in_2; 

assign gt_o    = in_1 >  in_2; 
assign lt_o    = in_1 <  in_2; 
assign geq_o   = in_1 >= in_2; 
assign leq_o   = in_1 <= in_2; 

endmodule
	
