import openai


def send_messages_to_gpt(messages):
    openai.api_key = # An OpenAI API key
    MODEL = "gpt-3.5-turbo"
    # An example of a system message that primes the assistant to give brief, to-the-point answers
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=0,
    )
    return response["choices"][0]["message"]["content"]


def table_to_text(table):
    messages = [
        {"role": "system", "content": "you are a middleschool physical teacher and try to describe this table to your students with very brief sentences by using distant step information."},
        {"role": "user", "content": "step	is_collision	kind	id_1	type_1	color_1	x_1	y_1	x_vel_1	y_vel_1	angle_1	id_2	type_2	color_2	x_2	y_2	x_vel_2	y_vel_2	angle_2; 78	TRUE	"
                                    "begin	0	boundary	Black	21.3333	-0.416667	0	0	0	5	circle	Blue	32	2.57833	0	-12.9033	0; 78	TRUE	begin	0	boundary	Black	"
                                    "21.3333	-0.416667	0	0	0	4	circle	Green	14.9333	1.245	0	-12.9033	0;79	TRUE	end	0	boundary	Black	21.3333	-0.416667	0	0	"
                                    "0	5	circle	Blue	32	2.60049	0	2.58067	0;79	TRUE	end	0	boundary	Black	21.3333	-0.416667	0	0	0	4	circle	Green	14.9333	1.26762	"
                                    "0	2.58067	0;110	TRUE	begin	0	boundary	Black	21.3333	-0.416667	0	0	0	5	circle	Blue	32	2.58361	0	-2.48267	0;110	TRUE	"
                                    "begin	0	boundary	Black	21.3333	-0.416667	0	0	0	4	circle	Green	14.9333	1.25074	0	-2.48267	0;112	TRUE	end	0	boundary	Black	"
                                    "21.3333	-0.416667	0	0	0	5	circle	Blue	32	2.59853	0	0.365867	0;112	TRUE	end	0	boundary	Black	21.3333	-0.416667	0	0	0	"
                                    "4	circle	Green	14.9333	1.26566	0	0.365867	0;117	TRUE	begin	0	boundary	Black	21.3333	-0.416667	0	0	0	5	circle	Blue	32	"
                                    "2.58818	0	-0.4508	0;117	TRUE	begin	0	boundary	Black	21.3333	-0.416667	0	0	0	4	circle	Green	14.9333	1.25532	0	-0.4508	0;137	"
                                    "TRUE	begin	5	circle	Blue	32	2.58833	0	0	0	6	User Circle	Red	36.6667	5.93351	0	-22.3767	0;138	TRUE	end	5	circle	Blue	31.9765	"
                                    "2.58833	-1.41076	1.34E-07	0.0060798	6	User Circle	Red	36.7988	5.78944	7.92529	-8.64408	-0.0701899;153	TRUE	begin	0	boundary	Black	"
                                    "21.3333	-0.416667	0	0	0	6	User Circle	Red	38.8787	3.16167	7.92529	-11.2574	-1.1739;154	TRUE	end	0	boundary	Black	21.3333	-0.416667	"
                                    "0	0	0	6	User Circle	Red	38.9197	3.18055	9.71523	2.25148	-1.18687;157	TRUE	begin	3	boundary	Black	43.0833	21.3333	0	0	0	6	User "
                                    "Circle	Red	39.505	3.29316	9.71523	1.59815	-1.37179;158	TRUE	end	3	boundary	Black	43.0833	21.3333	0	0	0	6	User Circle	Red	39.4827	3.32081	"
                                    "-1.94305	4.30486	-1.38053;211	TRUE	begin	0	boundary	Black	21.3333	-0.416667	0	0	0	6	User Circle	Red	37.7379	3.16167	-1.94305	"
                                    "-4.51514	-2.59673;213	TRUE	end	0	boundary	Black	21.3333	-0.416667	0	0	0	6	User Circle	Red	37.7403	3.18517	0.12908	0.739695	"
                                    "-2.59749;222	TRUE	begin	0	boundary	Black	21.3333	-0.416667	0	0	0	6	User Circle	Red	37.7596	3.17362	0.12908	-0.730305	-2.60361;794	"
                                    "TRUE	begin	4	circle	Green	14.9333	1.25532	0	0	0	5	circle	Blue	18.5107	2.58833	-1.20921	0	5.22222;795	TRUE	end	0	boundary	"
                                    "Black	21.3333	-0.416667	0	0	0	5	circle	Blue	18.4986	2.59395	-0.745785	0.331299	5.22615;799	TRUE	begin	0	boundary	Black	21.3333	"
                                    "-0.416667	0	0	0	5	circle	Blue	18.4523	2.59217	-0.644084	-0.212752	5.24034;"},
        {"role": "assistant", "content": "All three balls drop. The green and blue hit the ground first and are stationary, while the red falls and bumps into the blue ball. THe blue ball rolls and "
                                         "knocks the green ball and they touch."},
        {"role": "user", "content": table}
    ]
    return send_messages_to_gpt(messages)


def new_table_to_text(table):
    messages = [
        {"role": "user", "content": f"""I want you to summarize a table of physical salient events into explanations for the sequence of salient collisions. The following explanation is an example. 
        The table is  | step | is_collision | kind | id_1 | type_1 | color_1 | x_1 | y_1 | x_vel_1 | y_vel_1 | angle_1 | id_2 | type_2 | color_2 | x_2 | y_2 | x_vel_2 | y_vel_2 | angle_2 | | 
        -------- | ---------------- | -------- | -------- | ---------- | ----------- | --------- | --------- | ----------- | ----------- | ----------- | -------- | ----------- | ----------- | 
        ------- | ------- | ----------- | ----------- | ----------- | | 78 | TRUE             | begin    | 0        | boundary   | Black       | 21.3333   | -0.416667 | 0           | 0           | 
        0           | 4        | circle      | Green       | 8.53333 | 2.57833 | 0           | -12.9033    | 0           | | 78 | TRUE             | begin    | 0        | boundary   | Black       | 
        21.3333   | -0.416667 | 0           | 0           | 0           | 5        | circle      | Blue        | 36.2667 | 1.91167 | 0           | -12.9033    | 0           | | 79 | TRUE            
         | end      | 0        | boundary   | Black       | 21.3333   | -0.416667 | 0           | 0           | 0           | 5        | circle      | Blue        | 36.2667 | 1.93429 | 0           
         | 2.58067     | 0           | | 79 | TRUE             | end      | 0        | boundary   | Black       | 21.3333   | -0.416667 | 0           | 0           | 0           | 4        | circle 
              | Green       | 8.53333 | 2.60049 | 0           | 2.58067     | 0           | | 110 | TRUE             | begin    | 0        | boundary   | Black       | 21.3333   | -0.416667 | 0     
                    | 0           | 0           | 5        | circle      | Blue        | 36.2667 | 1.91741 | 0           | -2.48267    | 0           | | 110 | TRUE             | begin    | 0        
                    | boundary   | Black       | 21.3333   | -0.416667 | 0           | 0           | 0           | 4        | circle      | Green       | 8.53333 | 2.58361 | 0           | -2.48267  
                      | 0           | | 110 | TRUE             | begin    | 4        | circle     | Green       | 8.53333   | 2.58361   | 0           | -2.48267    | 0           | 6        | User 
                      Circle | Red         | 6.16667 | 8.38084 | 0           | -17.9667    | 0           | | 112 | TRUE             | end      | 0        | boundary   | Black       | 21.3333   | 
                      -0.416667 | 0           | 0           | 0           | 5        | circle      | Blue        | 36.2667 | 1.93233 | 0           | 0.365867    | 0           | | 112 | TRUE         
                          | end      | 0        | boundary   | Black       | 21.3333   | -0.416667 | 0           | 0           | 0           | 4        | circle      | Green       | 8.61241 | 
                          2.59668 | 1.47177     | 0.365864    | -0.0190248  | | 112 | TRUE             | end      | 4        | circle     | Green       | 8.61241   | 2.59668   | 1.47177     | 
                          0.365864    | -0.0190248  | 6        | User Circle | Red         | 6.00424 | 8.46559 | -4.26573    | 1.03595     | 0.057187    | | 117 | TRUE             | begin    | 0    
                              | boundary   | Black       | 21.3333   | -0.416667 | 0           | 0           | 0           | 5        | circle      | Blue        | 36.2667 | 1.92198 | 0           | 
                              -0.4508     | 0           | | 117 | TRUE             | begin    | 0        | boundary   | Black       | 21.3333   | -0.416667 | 0           | 0           | 0           
                              | 4        | circle      | Green       | 8.73506 | 2.58634 | 1.47177     | -0.450802   | -0.0665592  | | 142 | TRUE             | begin    | 1        | boundary   | 
                              Black       | -0.416667 | 21.3333   | 0           | 0           | 0           | 6        | User Circle | Red         | 3.82833 | 7.67709 | -4.26573    | -4.02738    | 
                              0.929929    | | 144 | TRUE             | end      | 1        | boundary   | Black       | -0.416667 | 21.3333   | 0           | 0           | 0           | 6        | 
                              User Circle | Red         | 3.85754 | 7.64025 | 0.853145    | -1.63128    | 0.93858     | | 178 | TRUE             | begin    | 4        | circle     | Green       | 
                              10.2287   | 2.58833   | 1.46675     | 0           | -0.645332   | 6        | User Circle | Red         | 4.34099 | 5.09614 | 0.853145    | -7.18462    | 1.14887     | 
                              | 181 | TRUE             | end      | 4        | circle     | Green       | 10.3526   | 2.58823   | 2.38571     | 3.52421E-09 | -0.669046   | 6        | User Circle | 
                              Red         | 4.31969 | 4.79549 | -0.421473   | -6.17809    | 1.1902      | | 189 | TRUE             | begin    | 0        | boundary   | Black       | 21.3333   | 
                              -0.416667 | 0           | 0           | 0           | 6        | User Circle | Red         | 4.26099 | 3.82833 | -0.421473   | -7.64809    | 1.30521     | | 190 | TRUE 
                                          | end      | 0        | boundary   | Black       | 21.3333   | -0.416667 | 0           | 0           | 0           | 6        | User Circle | Red         | 
                                          4.24667 | 3.85412 | -1.33557    | 1.52962     | 1.30896     | | 208 | TRUE             | begin    | 0        | boundary   | Black       | 21.3333   | 
                                          -0.416667 | 0           | 0           | 0           | 6        | User Circle | Red         | 3.82973 | 3.82833 | -1.33557    | -1.57372    | 1.41762     | 
                                          | 208 | TRUE             | begin    | 1        | boundary   | Black       | -0.416667 | 21.3333   | 0           | 0           | 0           | 6        | 
                                          User Circle | Red         | 3.82973 | 3.82833 | -1.33557    | -1.57372    | 1.41762     | | 210 | TRUE             | end      | 1        | boundary   | 
                                          Black       | -0.416667 | 21.3333   | 0           | 0           | 0           | 6        | User Circle | Red         | 3.84345 | 3.84144 | 0.267113    | 
                                          0.141989    | 1.41931     | | 820 | TRUE             | begin    | 4        | circle     | Green       | 31.835    | 2.58833   | 1.98076     | 0           | 
                                          -8.98287    | 5        | circle      | Blue        | 36.2667 | 1.92198 | 0           | 0           | 0           | | 821 | TRUE             | end      | 0  
                                                | boundary   | Black       | 21.3333   | -0.416667 | 0           | 0           | 0           | 4        | circle      | Green       | 31.8498 | 
                                                2.59712 | 0.914884    | 0.52287     | -8.98897    | | 828 | TRUE             | begin    | 0        | boundary   | Black       | 21.3333   | -0.416667 
                                                | 0           | 0           | 0           | 4        | circle      | Green       | 31.9509 | 2.58583 | 0.762085    | -0.512509   | -9.02931    | 

And the explanation is  "The balls drop down with the green and blue hitting the ground and stopping. The red falls onto the green on the left side propelling the green across the ground until it 
touches the blue ball." Now let's imitate the previous example and explain the following table. 

{table}

Now let's imitate my example and explain the above table. Read my example carefully, you do not need to tell me what happens at every step. And human will usually pay more attention to the green 
ball and the red ball. Make sure that you mention them in the summary. It would be best if you summarized the table briefly in no more than a hundred words. 

Recap the previous human explanation "The balls drop down with the green and blue hitting the ground and stopping. The red falls onto the green on the left side propelling the green across the 
ground until it touches the blue ball."
 """
         },
    ]
    return send_messages_to_gpt(messages)


def prompt_wrapper(prompt):
    messgaes = [
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt},
    ]
    return messgaes


if __name__ == "__main__":
    txt = table_to_text(
        "step	is_collision	kind	id_1	type_1	color_1	x_1	y_1	x_vel_1	y_vel_1	angle_1	id_2	type_2	color_2	x_2	y_2	x_vel_2	y_vel_2	angle_2;58	TRUE	begin	5	bar	Black	21.3333	2.90667	0	0	0	8	circle	Green	29.5737	4.41167	0	-9.63667	0;59	TRUE	end	5	bar	Black	21.3333	2.90667	0	0	0	8	circle	Green	29.5737	4.43133	0	1.92733	0;82	TRUE	begin	5	bar	Black	21.3333	2.90667	0	0	0	8	circle	Green	29.5737	4.41881	0	-1.82933	0;84	TRUE	end	5	bar	Black	21.3333	2.90667	0	0	0	8	circle	Green	29.5737	4.42937	0	0.2352	0;87	TRUE	begin	5	bar	Black	21.3333	2.90667	0	0	0	8	circle	Green	29.5737	4.42479	0	-0.2548	0;92	TRUE	begin	8	circle	Green	29.5737	4.42479	0	0	0	9	User Circle	Red	28.1667	10.1877	0	-15.0267	0;96	TRUE	end	8	circle	Green	29.6979	4.42117	0.967835	5.59E-09	-0.0666622	9	User Circle	Red	28.0237	10.2785	-2.38177	3.91035	0.0533922;163	TRUE	begin	5	bar	Black	21.3333	2.90667	0	0	0	9	User Circle	Red	25.3259	8.32833	-2.38177	-7.19632	1.167;165	TRUE	end	5	bar	Black	21.3333	2.90667	0	0	0	9	User Circle	Red	25.2703	8.36034	-3.217	1.27593	1.17813;181	TRUE	begin	5	bar	Black	21.3333	2.90667	0	0	0	9	User Circle	Red	24.4124	8.33036	-3.217	-1.3374	1.34952;547	TRUE	begin	1	boundary	Black	-0.416667	21.3333	0	0	0	9	User Circle	Red	5.00577	8.33833	-3.14925	0	5.2328;548	TRUE	end	1	boundary	Black	-0.416667	21.3333	0	0	0	9	User Circle	Red	5.01627	8.35402	0.629851	0.941391	5.23594;548	TRUE	end	5	bar	Black	21.3333	2.90667	0	0	0	9	User Circle	Red	5.01627	8.35402	0.629851	0.941391	5.23594;560	TRUE	begin	5	bar	Black	21.3333	2.90667	0	0	0	9	User Circle	Red	5.14224	8.32997	0.629851	-1.01861	5.27357;757	TRUE	begin	7	bar	Black	41.6765	-0.713897	0	0	-1.13446	8	circle	Green	40.1669	4.42167	0.932925	0	-9.75277;766	TRUE	end	7	bar	Black	41.6765	-0.713897	0	0	-1.13446	8	circle	Green	40.3024	4.4263	0.89735	-4.54E-09	-9.87798;771	TRUE	end	5	bar	Black	21.3333	2.90667	0	0	0	8	circle	Green	40.3774	4.42517	0.904465	-0.0350133	-9.9472;772	TRUE	begin	5	bar	Black	21.3333	2.90667	0	0	0	8	circle	Green	40.3924	4.42187	0.904465	-0.198347	-9.96113;789	TRUE	end	5	bar	Black	21.3333	2.90667	0	0	0	8	circle	Green	40.6841	4.3643	1.18958	-0.394682	-10.2365;790	TRUE	begin	5	bar	Black	21.3333	2.90667	0	0	0	8	circle	Green	40.7039	4.355	1.18958	-0.558016	-10.2558;791	TRUE	begin	4	bar	Purple	42.24	21.3333	0	0	1.5708	8	circle	Green	40.7246	4.34721	1.24387	-0.467384	-10.2763;793	TRUE	end	4	bar	Purple	42.24	21.3333	0	0	1.5708	8	circle	Green	40.7184	4.3497	-0.125236	0.0493825	-10.2771;799	TRUE	begin	4	bar	Purple	42.24	21.3333	0	0	1.5708	8	circle	Green	40.7215	4.34847	0.123602	-0.048595	-10.2802;"
    )
    print(txt)
