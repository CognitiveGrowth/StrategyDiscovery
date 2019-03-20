// based on normal dist with mean 0, std 40, truncated at +/-100
p_payoff_values = [0.00044358508219813293,0.00047204479573735711,0.00050201659714878169,0.00053355785421599963,0.00056672652412630348,0.00060158104117465539,0.00063818019591529583,0.00067658300569628819,0.00071684857654697039,0.00075903595642575297,0.00080320397987465771,0.00084941110416869017,0.00089771523709164796,0.00094817355651491972,0.00100084232200364472,0.00105577667872253458,0.00111303045396377617,0.00117265594667164503,0.00123470371038837960,0.00129922233010096679,0.00136625819352023735,0.00143585525737720851,0.00150805480937502945,0.00158289522648711873,0.00166041173034414203,0.00174063614050283381,0.00182359662643922713,0.00190931745915517634,0.00199781876333317504,0.00208911627101603147,0.00218322107782752085,0.00228013940278642534,0.00237987235279847652,0.00248241569293940504,0.00258775962366503305,0.00269588856610531412,0.00280678095661168574,0.00292040905173631890,0.00303673874482554345,0.00315572939540573454,0.00327733367253141283,0.00340149741325159599,0.00352815949732525119,0.00365725173929159998,0.00378869879896447364,0.00392241811137765753,0.00405831983716055476,0.00419630683426700478,0.00433627465191889754,0.00447811154755678627,0.00462169852751499917,0.00476690941205769672,0.00491361092532409220,0.00506166281063973218,0.00521091797155097229,0.00536122263883728658,0.00551241656365070425,0.00566433323681417387,0.00581680013420459753,0.00596963898801879025,0.00612266608360857462,0.00627569258144364522,0.00642852486364108976,0.00658096490437516061,0.00673281066336000461,0.00688385650147262801,0.00703389361746797025,0.00718271050461332609,0.00733009342596384166,0.00747582690687919899,0.00761969424328618351,0.00776147802408222627,0.00790096066598675387,0.00803792495905419901,0.00817215462098558505,0.00830343485830033028,0.00843155293236516785,0.00855629872822474477,0.00867746532412792468,0.00879484955961123971,0.00890825259997430827,0.00901748049496352462,0.00912234472948127845,0.00922266276413749760,0.00931825856348487795,0.00940896310980015964,0.00949461490031850569,0.00957506042587670698,0.00965015462898149469,0.00971976133939122210,0.00978375368538188413,0.00984201447895945332,0.00989443657338131242,0.00994092319146226526,0.00998138822325472276,0.01001575649182357385,0.01004396398596653535,0.01006595805887281446,0.01008169759185489553,0.01009115312244385221,0.01009430693628607378,0.01009115312244379496,0.01008169759185495278,0.01006595805887281446,0.01004396398596659086,0.01001575649182351833,0.00998138822325472276,0.00994092319146226526,0.00989443657338131242,0.00984201447895945332,0.00978375368538188413,0.00971976133939122210,0.00965015462898149469,0.00957506042587676250,0.00949461490031844844,0.00940896310980015964,0.00931825856348487795,0.00922266276413749760,0.00912234472948127845,0.00901748049496358013,0.00890825259997419551,0.00879484955961129522,0.00867746532412798019,0.00855629872822468752,0.00843155293236516785,0.00830343485830038579,0.00817215462098558505,0.00803792495905419901,0.00790096066598675387,0.00776147802408216989,0.00761969424328621126,0.00747582690687919899,0.00733009342596381303,0.00718271050461338247,0.00703389361746788525,0.00688385650147265576,0.00673281066335992048,0.00658096490437521699,0.00642852486364117390,0.00627569258144358971,0.00612266608360857462,0.00596963898801881887,0.00581680013420456891,0.00566433323681411749,0.00551241656365081614,0.00536122263883723020,0.00521091797155097229,0.00506166281063975994,0.00491361092532403669,0.00476690941205772534,0.00462169852751497141,0.00447811154755680015,0.00433627465191889754,0.00419630683426700478,0.00405831983716055476,0.00392241811137765753,0.00378869879896451571,0.00365725173929157222,0.00352815949732530713,0.00340149741325156824,0.00327733367253135646,0.00315572939540584730,0.00303673874482548751,0.00292040905173623477,0.00280678095661181888,0.00269588856610519486,0.00258775962366514537,0.00248241569293928535,0.00237987235279853247,0.00228013940278641841,0.00218322107782752779,0.00208911627101608047,0.00199781876333309785,0.00190931745915519737,0.00182359662643919200,0.00174063614050284443,0.00166041173034415959,0.00158289522648710117,0.00150805480937511011,0.00143585525737717338,0.00136625819352018466,0.00129922233010100908,0.00123470371038840411,0.00117265594667158887,0.00111303045396382539,0.00105577667872253111,0.00100084232200362369,0.00094817355651488286,0.00089771523709165847,0.00084941110416870773,0.00080320397987464545,0.00075903595642572131,0.00071684857654698969,0.00067658300569635845,0.00063818019591520801,0.00060158104117467295,0.00056672652412633861,0.00053355785421600396,0.00050201659714882289,0.00047204479573728685,0.00044358508219812328];

//#circleYellow {
//      width: 10px;
//      height: 10px;
//      -webkit-border-radius: 5px;
//      -moz-border-radius: 5px;
//      border-radius: 5px;
//      background: yellow;
//    }

group = _.sample([1,2])
if (group==1){
    var payoff_range1 = [0.01, 0.25];
    var payoff_range2 = [0.01, 9.99];
    block1_stakes = "low-stakes"
    block2_stakes = "high-stakes"
    Block1_stakes = "Low-stakes"
    Block2_stakes = "High-stakes"
}
else{
    var payoff_range2 = [0.01, 0.25];
    var payoff_range1 = [0.01, 9.99];
    block1_stakes = "high-stakes"
    block2_stakes = "low-stakes"
    Block1_stakes = "High-stakes"
    Block2_stakes = "Low-stakes"
}
instructions = '1. Some games have low-stakes and some games have high-stakes. For low-stakes games, the range of possible earnings is $0.01 at least to $0.25 at most. For high-stakes games, the range of possible winnings is $0.01 at least to $9.99 at most. The last example was high-stakes. <b>The first 10 games will be <u>'+block1_stakes+'</u> and the last 10 games will be <u>'+block2_stakes+'</u></b>. Your bonus payment will be determined by selecting one of your winnings from '+block1_stakes+' games and one of your winnings from '+block2_stakes+' games, and averaging the two.'
$("#instructions_text").html(" hi! ");

nr_blocks=2;
minutes_per_block=2.5;
max_bonus=2; 

priming={
    p_compensatory: 1,
    p_high_stakes:  0.5,
    p_all_good: 0,
    p_all_bad: 0,
    p_large: 1,
    with_feedback: [true],
    low_stakes: 10,
    high_stakes: 100,
    many_outcomes: 4,
    few_outcomes: 4,
    many_gambles: 5,
    few_gambles: 5,
    nr_good_trials_at_the_start: 0
}

pre_and_posttest={    
    p_compensatory: 1,
    p_high_stakes:  0,
    p_all_good: 0,
    p_all_bad: 0.75,
    p_large: 1,
    with_feedback: [true],
    low_stakes: 10,
    high_stakes: 20,
    many_outcomes: 4,
    few_outcomes: 2,
    many_gambles: 5,
    few_gambles: 4,
    nr_good_trials_at_the_start: 4,
}

training={
    p_compensatory: 1,
    p_high_stakes:  0,
    p_all_good: 0,
    p_all_bad: 0.75,
    p_large: 1,
    with_feedback: [true],
    low_stakes: 10,
    high_stakes: 20,
    many_outcomes: 4,
    few_outcomes: 2,
    many_gambles: 5,
    few_gambles: 4,
    nr_good_trials_at_the_start: 4
}

img_by_problem_type=["hexagon.png","OrangeDiamond.png","BlueMountain.png","PurpleSun.png"];

high_range=Math.max(Math.max(training.high_range,pre_and_posttest.high_stakes),priming.high_stakes);
low_range=Math.min(Math.min(training.low_stakes,pre_and_posttest.low_stakes),priming.low_stakes);
with_feedback=[priming.with_feedback,priming.with_feedback].concat(pre_and_posttest.with_feedback).concat(training.with_feedback).concat(pre_and_posttest.with_feedback);

cell_height=50;
cell_width=100;

preset100_game_rewards=new Array();  preset100_game_rewards.push([0.02,0.06,0.08,0.15,0.05,0.16,0.07,0.15,0.14,0.04,0.12,0.16,0.1,0.08,0.2,0.15,0.21,0.19,0.18,0.06,0.19,0.15,0.02,0.05,0.11,0.14,0.11,0.16,0.19,0.09,0.23,0.12,0.1,0.14,0.06,0.11,0.15,0.18,0.17,0.11,0.08,0.01,0.2,0.12,0.21,0.11,0.22,0.14,0.1,0.07,5.97,8.47,3.4,2.65,0.95,2.54,7.32,4.41,8.39,9.88,3.25,2.44,7.17,5.93,3.89,0.94,3.54,7.04,6.26,1.63,0.72,5.71,6.19,3.74,5.16,7.74,1.32,8.51,5.28,3.95,7.78,6.2,2.67,2.88,1.89,5.11,3.55,1.92,6.18,7.91,7.29,7.11,6.38,5.63,2.09,4.36,5.92,8.02,5.77,5.12]); preset100_game_rewards.push([0.1,0.22,0.23,0.08,0.17,0.21,0.16,0.06,0.15,0.14,0.04,0.11,0.22,0.23,0.21,0.17,0.14,0.09,0.08,0.1,0.11,0.08,0.18,0.08,0.18,0.12,0.06,0.14,0.12,0.06,0.06,0.13,0.1,0.07,0.11,0.17,0.2,0.08,0.04,0.13,0.08,0.19,0.14,0.17,0.18,0.22,0.12,0.16,0.1,0.07,8.23,2.82,2.14,1.66,2.78,7.44,4.38,7.98,1.9,3.37,0.97,7.36,3.46,4.61,7.03,9.58,3.18,7.27,0.19,4.7,5.25,4,1.85,4.35,3.84,3.22,1.79,2.86,9.9,8.96,7.83,4.19,1.86,2.6,1.68,2.17,2.94,6.77,4.93,1.03,4.26,5.78,4.4,2.91,7.52,8.7,9.68,3.92,3.99,1.51]); preset100_game_rewards.push([0.05,0.11,0.11,0.15,0.08,0.19,0.11,0.09,0.11,0.09,0.15,0.08,0.11,0.16,0.23,0.12,0.05,0.13,0.04,0.22,0.24,0.16,0.09,0.17,0.17,0.08,0.11,0.08,0.08,0.16,0.08,0.12,0.17,0.05,0.17,0.16,0.21,0.14,0.18,0.2,0.03,0.1,0.16,0.14,0.18,0.08,0.18,0.16,0.03,0.17,1.24,7.33,9.33,4.12,3.25,2.24,4.99,5.03,3.96,6.84,8.64,6.46,5.58,6.44,1.42,9.58,6.26,1.99,5.42,3.32,9.97,7.87,1.16,1.03,2.97,6.43,4.74,5.22,6.81,8.28,6.34,3.44,1.79,7.89,4.82,0.1,0.08,0.18,6.72,5.87,4.93,4.85,2.78,6.84,9.74,5.94,5.15,3.11,6.61,5.2]); preset100_game_rewards.push([0.19,0.12,0.15,0.12,0.07,0.12,0.2,0.11,0.2,0.06,0.03,0.09,0.07,0.14,0.23,0.06,0.18,0.2,0.19,0.17,0.23,0.16,0.13,0.15,0.05,0.02,0.08,0.14,0.18,0.11,0.13,0.14,0.08,0.18,0.23,0.1,0.14,0.2,0.09,0.15,0.08,0.03,0.12,0.13,0.07,0.2,0.2,0.21,0.13,0.05,7.62,4.03,2.59,6.28,5.38,4.32,7.76,0.94,5.78,6.02,8.18,5.15,3.93,6.58,3.85,1.68,3.39,4.87,8.23,2.64,4.85,3.26,4.28,5.02,6.89,3.71,6.52,3.29,1.61,6.18,3.49,8.27,5.75,4.77,0.64,9.47,3.13,0.58,3.3,7.52,3.96,4.33,4.02,6.5,6.72,4.41,2.59,4.49,4.81,6.71]); preset100_game_rewards.push([0.09,0.13,0.14,0.15,0.16,0.21,0.2,0.16,0.03,0.18,0.12,0.11,0.15,0.22,0.18,0.06,0.12,0.03,0.05,0.09,0.21,0.12,0.07,0.22,0.02,0.14,0.17,0.11,0.08,0.12,0.11,0.05,0.04,0.03,0.15,0.16,0.15,0.12,0.04,0.12,0.06,0.13,0.21,0.11,0.17,0.14,0.04,0.14,0.2,0.08,4.62,7.07,4.55,3.9,7.09,2.76,5,6.07,2.24,5.72,2.13,7.43,3.7,0.77,5.57,3.99,3.52,6.53,3.15,8.47,8.58,8.95,2.66,2.48,7.92,7.81,9.55,7.8,2.79,7.02,9.25,6.51,3.59,7.12,6.72,2.73,2.15,7.09,6.34,6.88,3.34,7.33,3.28,2.65,2.88,8.37,5.89,4.13,8.16,2.69]); preset100_game_rewards.push([0.05,0.09,0.16,0.08,0.13,0.05,0.19,0.04,0.13,0.1,0.21,0.11,0.15,0.12,0.14,0.22,0.1,0.17,0.05,0.03,0.14,0.04,0.12,0.07,0.11,0.19,0.11,0.13,0.2,0.22,0.05,0.1,0.09,0.01,0.14,0.08,0.21,0.1,0.16,0.07,0.11,0.15,0.18,0.07,0.09,0.03,0.04,0.04,0.08,0.18,4.29,2.75,1.09,4.24,3.77,9.01,5.94,7.96,5.8,2.47,6.22,1.32,2.49,6.66,3.85,3.26,5.47,6.34,1.79,3.24,2.57,9.97,3.25,7.72,3.02,4.87,1.49,7.12,8.92,4.65,4.1,4.65,3.3,4.23,5.82,6.4,9.59,4.34,6.06,7.18,8.54,7.89,4.56,5.91,5.9,5.83,7.38,7.86,4.87,8.45]); preset100_game_rewards.push([0.06,0.18,0.07,0.22,0.05,0.2,0.16,0.14,0.04,0.07,0.16,0.06,0.08,0.2,0.22,0.15,0.15,0.2,0.16,0.08,0.16,0.13,0.08,0.07,0.14,0.09,0.11,0.09,0.22,0.04,0.11,0.25,0.12,0.18,0.21,0.16,0.2,0.1,0.22,0.12,0.11,0.13,0.23,0.23,0.08,0.13,0.16,0.08,0.11,0.15,5.98,1.54,2.85,8.17,4.83,8.48,2.36,3.08,4.1,6.83,0.77,0.71,5.53,0.58,2.09,9.98,1.42,1.47,3.82,4.54,9.15,6.35,7.93,3.91,6.81,5,3.36,2.91,6.09,4.21,8.16,4.01,1.49,2.87,3.47,3.36,4.77,4.61,5.41,6.1,6.49,5.26,6.79,4.09,5.87,0.58,2.85,3.71,2.84,7.56]); preset100_game_rewards.push([0.03,0.2,0.19,0.12,0.1,0.14,0.15,0.13,0.03,0.09,0.15,0.14,0.22,0.17,0.2,0.19,0.03,0.08,0.09,0.24,0.12,0.17,0.15,0.06,0.16,0.2,0.14,0.19,0.05,0.11,0.17,0.16,0.18,0.19,0.16,0.14,0.1,0.1,0.22,0.17,0.21,0.17,0.06,0.21,0.07,0.17,0.19,0.06,0.08,0.19,8.29,4.68,2.69,4.84,7.13,1.82,8.4,7.18,0.94,4.66,5.88,3.96,9.77,1.24,1.42,7.2,3.13,5.29,4.19,8.67,5.48,4.81,4.16,9.52,6.93,6.79,2.67,7.85,3.46,7.39,4.02,2.89,1.18,3.29,2.81,4.31,4.35,2.63,7.2,5.88,0.79,5.95,6.89,5.16,6.25,9.58,1.22,5.5,5.38,1.26]); preset100_game_rewards.push([0.12,0.17,0.17,0.2,0.16,0.1,0.22,0.19,0.07,0.13,0.15,0.14,0.14,0.2,0.22,0.17,0.07,0.14,0.22,0.07,0.1,0.17,0.12,0.1,0.12,0.01,0.12,0.11,0.04,0.19,0.17,0.2,0.07,0.23,0.17,0.11,0.12,0.05,0.14,0.14,0.12,0.18,0.09,0.03,0.09,0.02,0.18,0.2,0.05,0.17,4.65,3.16,7.17,2.23,8.96,8.34,3.61,6.38,5.02,3.57,7.57,8.36,5.93,4.91,3.42,6.69,1.91,2.16,3.39,5.47,9.36,5.73,3.03,3.82,3.64,7.24,2.3,5.63,6.68,4.82,5.78,4.13,8.77,2.97,5.64,7.26,5.83,2.6,2.38,5.55,3.44,7.87,5.4,0.83,4.49,2.24,5.64,3.12,5.12,6.81]); preset100_game_rewards.push([0.22,0.05,0.09,0.22,0.24,0.04,0.24,0.06,0.1,0.01,0.1,0.15,0.05,0.03,0.22,0.21,0.04,0.18,0.23,0.1,0.02,0.2,0.24,0.12,0.11,0.13,0.06,0.19,0.19,0.13,0.17,0.2,0.12,0.07,0.08,0.1,0.15,0.2,0.04,0.08,0.16,0.07,0.16,0.18,0.01,0.02,0.17,0.17,0.11,0.06,5.55,0.97,4.91,5.57,5.39,6.23,8.45,1.11,2.49,1.89,5.25,4.6,5.47,5.63,3.49,3.6,7.14,9.06,2.75,1.74,2.36,5.74,5.34,4.46,5.33,2.38,3.39,4.76,6.44,4.44,4.37,4.82,6.89,3.82,4.93,6.36,1.82,4.89,3.93,1.78,2.71,3.44,5.87,2.41,6.66,8.49,6.3,4.78,5.88,6.02]); preset100_game_rewards.push([0.08,0.09,0.16,0.16,0.2,0.07,0.17,0.21,0.21,0.23,0.11,0.18,0.13,0.17,0.25,0.18,0.14,0.15,0.17,0.11,0.19,0.1,0.14,0.11,0.08,0.03,0.14,0.08,0.18,0.18,0.13,0.14,0.15,0.1,0.04,0.17,0.05,0.13,0.11,0.03,0.16,0.17,0.16,0.17,0.16,0.09,0.17,0.04,0.15,0.13,1.01,1.44,8.03,7.45,2.27,3.89,4.23,4.77,5.54,8.02,3.43,6.91,2.25,0.45,1.78,6.35,3.81,4.67,9.2,1.77,9.25,5.66,3.93,2.55,8.47,4.86,7.85,5.73,3.71,1.73,3.5,8.89,4.16,9.51,5.88,8.48,2.79,2.52,9.67,3,8.71,7,9.63,5.99,2.68,4.24,3.82,4.63,4.67,6.85]); preset100_game_rewards.push([0.02,0.17,0.14,0.25,0.19,0.08,0.18,0.14,0.1,0.07,0.22,0.2,0.2,0.21,0.05,0.06,0.13,0.19,0.13,0.11,0.1,0.06,0.1,0.19,0.11,0.11,0.09,0.11,0.18,0.2,0.09,0.09,0.14,0.08,0.07,0.15,0.19,0.11,0.04,0.14,0.17,0.17,0.14,0.22,0.02,0.21,0.18,0.15,0.22,0.2,2.98,5.2,6.07,3.35,8.34,6.94,7.02,7.21,6.54,4.52,6.13,2.69,8.04,3.38,8.76,1.28,4.58,7.92,5.7,1.06,3.72,5.51,5.82,5.65,3.82,3.87,3.89,4.36,3.55,7.35,3.65,9.25,3.25,9.45,1.19,2.98,3.6,8.42,4.79,4.69,2.86,6.97,0.42,6.15,2.62,4.63,9.13,5.83,7.45,3.83]); preset100_game_rewards.push([0.07,0.16,0.09,0.12,0.19,0.16,0.12,0.09,0.13,0.19,0.11,0.23,0.14,0.17,0.12,0.19,0.15,0.11,0.24,0.15,0.08,0.16,0.01,0.16,0.16,0.06,0.14,0.19,0.05,0.12,0.2,0.17,0.22,0.19,0.21,0.09,0.19,0.09,0.23,0.12,0.12,0.12,0.12,0.07,0.06,0.16,0.2,0.07,0.2,0.13,8.74,6.65,6.49,5.55,7.96,3.11,4.36,4.82,4.33,4.58,3.52,5,7.17,4.94,8.52,0.52,6.9,5.41,3.44,7.43,4.76,4.81,4.05,6.25,8.88,6.17,7.4,4.99,0.25,9.66,5.34,0.96,3,8.27,6.58,4.68,4.35,2.23,7.96,4.52,7.09,5.22,4.93,3.67,7.58,4.53,6.51,9.55,7.64,2.04]); preset100_game_rewards.push([0.12,0.24,0.03,0.11,0.23,0.22,0.12,0.14,0.08,0.05,0.06,0.09,0.2,0.07,0.09,0.04,0.2,0.07,0.18,0.16,0.14,0.12,0.14,0.05,0.08,0.23,0.12,0.13,0.1,0.12,0.07,0.09,0.18,0.2,0.12,0.15,0.19,0.04,0.13,0.21,0.08,0.12,0.15,0.12,0.25,0.15,0.15,0.15,0.17,0.2,8.92,9.25,2.28,3.39,4.55,6.18,9.23,4.9,9.65,0.56,3.61,1.64,0.39,3.12,6.88,9.84,8.39,7.56,3.56,3.25,6.74,7.59,2.14,0.76,7.83,4.92,8.55,0.21,8.07,2.04,1.1,6.04,9.34,5.42,2.91,5.77,6.66,4.37,3.99,8.75,3.94,6.26,4.29,6.12,4.86,5.29,3.36,5.01,1.41,5.15]); preset100_game_rewards.push([0.18,0.14,0.13,0.19,0.18,0.03,0.11,0.16,0.14,0.08,0.06,0.09,0.02,0.08,0.02,0.07,0.05,0.15,0.21,0.04,0.05,0.04,0.2,0.19,0.12,0.19,0.04,0.21,0.21,0.11,0.21,0.11,0.17,0.17,0.2,0.18,0.2,0.15,0.09,0.15,0.24,0.11,0.23,0.06,0.21,0.09,0.03,0.13,0.16,0.08,1.93,8.41,2.24,6.15,9.77,7.6,1.69,5.63,7.94,5.57,5.54,8.24,7.48,7.83,3.98,4.59,7.48,6.95,5.45,4.05,7.14,6.14,6.53,2.96,4.51,3.15,4.18,1.1,6.52,6.67,4.29,8.8,1.75,3.22,6.88,8.38,4.91,6.54,0.82,5.71,9.61,3.72,2.96,5.67,0.33,6.08,6.72,6.87,2.86,0.77]); preset100_game_rewards.push([0.19,0.03,0.11,0.22,0.13,0.17,0.16,0.13,0.03,0.2,0.07,0.18,0.25,0.2,0.05,0.13,0.14,0.06,0.2,0.05,0.21,0.25,0.14,0.16,0.16,0.12,0.15,0.23,0.13,0.08,0.11,0.12,0.02,0.24,0.19,0.13,0.18,0.11,0.1,0.05,0.22,0.17,0.08,0.15,0.16,0.18,0.13,0.19,0.24,0.18,2.81,7.04,5.15,3.53,5.61,3.71,3.23,7.2,7.8,8.71,4.65,5.23,1.53,1.4,7.53,4.33,1.27,5.35,5.72,4.33,7.36,5.62,8.89,5.49,5.38,9.58,0.46,2.64,2.17,2.62,1.52,2.84,2.03,3.81,4.96,2,4.62,7.27,2.25,1.25,8.74,9.35,2.23,9.08,4.49,4.31,1,6.07,1.79,1.29]); preset100_game_rewards.push([0.1,0.23,0.24,0.15,0.06,0.22,0.11,0.11,0.06,0.08,0.19,0.2,0.2,0.14,0.07,0.08,0.08,0.07,0.1,0.2,0.07,0.2,0.24,0.13,0.05,0.16,0.09,0.11,0.15,0.15,0.17,0.11,0.06,0.03,0.13,0.06,0.22,0.22,0.01,0.06,0.1,0.18,0.17,0.14,0.24,0.16,0.18,0.14,0.08,0.04,4.79,1.31,4.83,5.47,8.97,5.31,3.54,7.79,4.1,8.98,8.68,2.67,5.91,2.04,6.94,3.41,5.41,1.7,8.61,1.96,2.91,3.6,6.73,2.32,9.21,2.44,4.32,4.86,8,2.43,3.77,4.5,4.01,4.18,4.06,6.14,9.06,4.71,1.43,6.22,7.3,7.04,8.29,3.93,3.44,5.59,8.86,1.32,4.9,6.67]); preset100_game_rewards.push([0.18,0.14,0.17,0.08,0.19,0.13,0.12,0.14,0.05,0.21,0.11,0.13,0.08,0.22,0.23,0.09,0.16,0.11,0.14,0.13,0.2,0.11,0.13,0.06,0.23,0.15,0.13,0.11,0.13,0.06,0.05,0.11,0.16,0.21,0.12,0.09,0.07,0.09,0.06,0.08,0.12,0.22,0.11,0.1,0.08,0.17,0.12,0.02,0.04,0.21,6.01,3.23,2.39,2.82,8.09,9.83,3.86,1.19,9.82,7.3,9.8,1.92,7.74,7.87,2.68,2.69,7.63,4.81,4.63,6.68,4.13,6.94,7.28,7.63,3.79,5.2,2.27,7.57,8.5,7.24,7.63,4.81,8.67,2.52,5.77,7.53,2.02,3.86,2.43,6.17,3.82,6.38,4.74,7.57,6,3.3,8.25,5.58,7.89,1.6]); preset100_game_rewards.push([0.18,0.04,0.14,0.12,0.17,0.04,0.21,0.1,0.15,0.07,0.05,0.05,0.15,0.1,0.14,0.18,0.04,0.19,0.11,0.03,0.09,0.04,0.22,0.18,0.13,0.16,0.15,0.1,0.16,0.15,0.13,0.08,0.1,0.19,0.17,0.04,0.23,0.14,0.15,0.07,0.06,0.08,0.04,0.11,0.24,0.05,0.17,0.08,0.05,0.15,7.29,7.09,2.78,6.99,6.52,1.09,3.72,2.56,1.59,1.41,7.31,6.03,1.59,6.09,5.16,0.89,6.12,6.84,8.16,6.28,7.03,3.42,2.75,0.35,4.77,8.45,3.18,7.15,5.35,6.32,3.05,3.19,5.48,3.48,1.03,4.7,4.04,8.83,5.42,8.38,1.83,7.42,3.72,5.62,4.73,3.91,8.82,8.22,6.3,1.63]); preset100_game_rewards.push([0.13,0.13,0.06,0.14,0.21,0.22,0.06,0.12,0.08,0.04,0.07,0.16,0.02,0.13,0.19,0.11,0.13,0.02,0.11,0.1,0.02,0.11,0.23,0.04,0.08,0.07,0.18,0.18,0.15,0.13,0.11,0.05,0.2,0.07,0.2,0.12,0.17,0.12,0.17,0.16,0.2,0.06,0.14,0.15,0.16,0.09,0.16,0.23,0.19,0.15,6.55,7.67,7.43,9.09,2.37,3.87,2.06,6.05,4.84,5.09,3.42,2.88,6.73,6.31,4.6,3.28,1.81,1.21,8.36,9.52,8.48,3.6,0.6,8.21,9.29,2.58,8.62,0.31,1.81,7.33,0.82,0.46,1.04,3.16,4.24,1.99,2.21,4.81,7.07,7.26,1.17,6.47,2.39,2.51,0.87,4.21,6.13,5.74,3.05,0.54]); preset100_game_rewards.push([0.15,0.18,0.15,0.2,0.13,0.12,0.16,0.12,0.09,0.22,0.16,0.07,0.07,0.17,0.02,0.12,0.08,0.14,0.19,0.19,0.23,0.24,0.18,0.24,0.03,0.21,0.21,0.03,0.08,0.16,0.17,0.17,0.19,0.21,0.16,0.18,0.15,0.12,0.05,0.13,0.05,0.07,0.15,0.22,0.09,0.1,0.08,0.1,0.14,0.18,4.48,1.55,5.61,0.79,9.08,7.81,0.61,3.13,8.58,6.3,4.37,7.54,3.86,5.4,0.82,2.91,6.34,8.92,7.19,7.64,0.38,6.26,4.09,6.51,4.59,2.29,2.94,2.34,3.32,5.82,4.95,7.31,6.3,9.41,5.62,1.53,7.62,5.31,9.93,4.32,5.37,5.89,0.86,2.61,2.63,4.74,4.12,5.54,5.2,0.89]); preset100_game_rewards.push([0.08,0.16,0.09,0.21,0.16,0.16,0.17,0.2,0.13,0.12,0.05,0.08,0.11,0.07,0.12,0.16,0.19,0.12,0.23,0.2,0.09,0.15,0.11,0.15,0.17,0.09,0.1,0.04,0.19,0.08,0.16,0.19,0.23,0.14,0.1,0.24,0.25,0.17,0.04,0.11,0.13,0.12,0.06,0.13,0.17,0.18,0.21,0.15,0.1,0.06,9.09,6.28,3.01,2.47,8.52,7.76,7.94,2.67,8.96,2.79,7.14,1.43,5.1,6.34,7.43,7.61,6.09,6.63,3.14,3.95,5.84,1.09,4.95,4.09,3.55,4.2,6.02,6.38,5.89,4.75,8.46,5.12,4.37,1.89,0.22,7.09,0.78,5.47,9.97,6.4,2.04,5.94,5.36,4.59,7.63,9.25,7.5,4.62,5.28,4.2]); preset100_game_rewards.push([0.17,0.1,0.17,0.16,0.18,0.14,0.22,0.09,0.08,0.1,0.12,0.02,0.1,0.12,0.1,0.1,0.12,0.04,0.14,0.09,0.04,0.14,0.21,0.24,0.24,0.14,0.08,0.14,0.06,0.15,0.02,0.2,0.18,0.09,0.1,0.17,0.17,0.01,0.06,0.11,0.19,0.02,0.12,0.03,0.08,0.05,0.09,0.03,0.18,0.08,1.98,2.99,5.69,2.31,1.84,7,7.02,2.05,7.49,3.58,2.83,5.93,6.8,6.33,7.31,5.48,3.7,5.53,2.27,5.24,6.37,4.7,6.55,2.1,8.48,1.81,2.09,5.91,1.86,4.96,6.15,1.87,0.5,5.73,1.89,5.94,5.9,8.91,3.97,8.88,1.98,4.84,5.18,1.2,1.46,5.33,7.75,3.36,6.58,5.58]); preset100_game_rewards.push([0.23,0.17,0.09,0.09,0.2,0.16,0.15,0.2,0.16,0.09,0.14,0.16,0.17,0.01,0.13,0.02,0.09,0.06,0.16,0.17,0.09,0.05,0.18,0.11,0.1,0.24,0.17,0.12,0.09,0.2,0.2,0.07,0.17,0.21,0.12,0.23,0.14,0.14,0.1,0.13,0.1,0.07,0.16,0.18,0.18,0.09,0.16,0.13,0.15,0.09,6.38,4.33,6.66,1.36,1.06,2.77,6.73,5.91,1.94,9.87,7.59,6.77,2.11,4.75,6.36,8.81,4.25,4.35,7.83,1.85,7.56,1.56,6.81,9.56,4.66,5.9,4.16,3.08,6.44,2.68,6.73,1.91,5.19,6.1,4.63,3.03,1.77,3.37,5.79,5.28,8.39,1.39,6.28,4.42,3.24,1.67,5.39,5.75,1.34,3.29]); preset100_game_rewards.push([0.11,0.16,0.17,0.16,0.12,0.16,0.11,0.06,0.01,0.18,0.09,0.15,0.14,0.11,0.23,0.11,0.06,0.04,0.18,0.15,0.14,0.11,0.07,0.13,0.22,0.12,0.08,0.13,0.07,0.17,0.19,0.21,0.22,0.08,0.16,0.17,0.07,0.05,0.03,0.18,0.18,0.16,0.23,0.17,0.24,0.19,0.11,0.07,0.14,0.02,4.5,1.28,3.22,3.11,3.62,7.48,4.28,4.15,3.92,3.61,3.91,1.1,7.09,6.81,5.75,6.78,4.91,1.42,9.11,2.91,3.71,4.05,6.9,8.08,2.28,9.47,3.95,6,5.29,1.71,3.18,7,4.46,1.69,5.2,5.37,4.45,8.65,8.9,4.23,1.63,4.52,1.08,1.16,5.72,7.75,3.9,5.59,2.2,3.97]); preset100_game_rewards.push([0.13,0.14,0.19,0.22,0.19,0.18,0.24,0.12,0.21,0.18,0.12,0.1,0.2,0.05,0.18,0.21,0.18,0.06,0.12,0.2,0.18,0.22,0.12,0.14,0.06,0.1,0.19,0.05,0.04,0.24,0.23,0.07,0.21,0.19,0.01,0.08,0.06,0.14,0.1,0.08,0.14,0.17,0.08,0.07,0.11,0.24,0.25,0.13,0.18,0.12,3.08,9.25,3.6,4.52,0.7,3.83,8.5,9.55,0.76,3.51,6.6,3.47,7.57,8.3,9,0.6,0.8,5.69,0.69,0.64,4.02,2.65,1.07,0.89,3.95,8.69,2.34,7.58,3.29,6.22,8.15,2.08,0.43,4.58,4.1,2.51,7.63,6.85,4.27,1.08,1.8,3.41,2.81,8.17,7.53,5.64,7.48,2.34,4.69,3.54]); preset100_game_rewards.push([0.16,0.04,0.1,0.15,0.08,0.13,0.11,0.09,0.09,0.04,0.13,0.09,0.08,0.12,0.06,0.13,0.05,0.19,0.18,0.1,0.2,0.22,0.14,0.03,0.14,0.2,0.13,0.04,0.2,0.04,0.13,0.13,0.09,0.05,0.17,0.13,0.14,0.07,0.03,0.12,0.09,0.14,0.11,0.25,0.09,0.18,0.1,0.07,0.07,0.19,5.26,6.99,8.48,5.21,3.27,0.31,0.13,4.46,2.82,7.13,1.41,5.78,6.62,6.3,9.88,7.91,7.46,5.92,3.38,2.21,2.93,4.29,2.2,1.16,2.95,2.45,3.36,3.85,8.96,1.58,6.29,5.06,0.13,1.23,3.36,3.9,4.39,1.47,7.39,0.7,1.97,2.82,5.95,8.27,7.25,2.81,3.23,1.7,6.92,6.37]); preset100_game_rewards.push([0.07,0.05,0.17,0.19,0.19,0.14,0.17,0.16,0.05,0.15,0.02,0.12,0.11,0.17,0.24,0.19,0.09,0.09,0.03,0.15,0.23,0.2,0.14,0.01,0.05,0.16,0.19,0.05,0.08,0.12,0.13,0.2,0.19,0.09,0.09,0.06,0.21,0.08,0.15,0.15,0.16,0.04,0.15,0.1,0.22,0.12,0.16,0.19,0.19,0.09,4.47,2.35,3.28,3.31,0.14,3.13,2.63,9.22,6.17,2.78,1.96,6.1,8.34,9.27,3.78,3.87,7.59,4.12,9.35,1.85,0.09,5.33,5.26,4.1,5.67,6.94,2.27,6.26,7.12,9.11,5.32,2.86,6.06,3.5,6.44,2.79,2.86,5.12,5.81,3.39,7.71,3.73,9.65,1.11,3.01,3.02,1.98,6.64,5.01,4.95]);

preset100_probabilities=new Array();  preset100_probabilities.push([0.06,0.87,0.01,0.05,0.85,0.06,0.09,0.08,0.06,0.05,0.04,0.08,0.87,0.86,0.08,0.09,0.01,0.02,0.02,0.01,0.03,0.01,0.85,0.86,0.01,0.36,0.26,0.31,0.16,0.15,0.24,0.17,0.21,0.15,0.23,0.26,0.33,0.3,0.35,0.11,0.12,0.3,0.33,0.39,0.21,0.33,0.34,0.26,0.36,0.31,0.86,0.01,0.07,0.01,0.85,0.04,0.85,0.04,0.11,0.02,0.06,0.02,0.88,0.01,0.86,0.01,0.04,0.07,0.1,0.05,0.03,0.02,0.03,0.04,0.86,0.15,0.16,0.19,0.23,0.25,0.2,0.15,0.24,0.2,0.25,0.31,0.34,0.31,0.28,0.3,0.21,0.33,0.2,0.14,0.34,0.31,0.3,0.25,0.34,0.21]); preset100_probabilities.push([0.06,0.02,0.02,0.07,0.1,0.89,0.03,0.04,0.03,0.03,0.05,0.02,0.08,0.03,0.03,0.05,0.02,0.85,0.92,0.1,0.01,0.89,0.02,0.01,0.08,0.17,0.2,0.22,0.28,0.32,0.22,0.35,0.32,0.26,0.25,0.27,0.32,0.22,0.15,0.38,0.34,0.29,0.18,0.18,0.27,0.18,0.12,0.28,0.26,0.27,0.01,0.01,0.01,0.06,0.07,0.07,0.03,0.02,0.85,0.05,0.03,0.06,0.05,0.1,0.01,0.89,0.07,0.87,0.01,0.89,0.85,0.08,0.86,0.08,0.08,0.38,0.11,0.32,0.3,0.24,0.29,0.21,0.23,0.16,0.26,0.35,0.11,0.2,0.28,0.18,0.28,0.2,0.38,0.34,0.23,0.27,0.17,0.27,0.23,0.27]); preset100_probabilities.push([0.03,0.1,0.11,0.86,0.02,0.04,0.01,0.87,0.86,0.07,0.89,0.89,0.02,0.04,0.87,0.01,0.88,0.05,0.02,0.04,0.92,0.09,0.03,0.05,0.86,0.12,0.27,0.18,0.23,0.21,0.21,0.31,0.34,0.26,0.3,0.3,0.13,0.27,0.13,0.34,0.21,0.13,0.11,0.15,0.2,0.11,0.22,0.24,0.19,0.19,0.12,0.01,0.9,0.02,0.07,0.03,0.11,0.05,0.02,0.05,0.01,0.89,0.05,0.87,0.02,0.08,0.88,0.03,0.01,0.05,0.06,0.03,0.04,0.02,0.05,0.14,0.34,0.23,0.24,0.28,0.22,0.3,0.24,0.26,0.31,0.15,0.27,0.17,0.19,0.2,0.29,0.24,0.3,0.35,0.2,0.14,0.24,0.24,0.11,0.19]); preset100_probabilities.push([0.85,0.01,0.86,0.02,0.03,0.01,0.87,0.01,0.05,0.85,0.02,0.01,0.03,0.07,0.02,0.85,0.09,0.08,0.04,0.85,0.04,0.01,0.1,0.08,0.05,0.35,0.27,0.29,0.33,0.32,0.33,0.17,0.13,0.33,0.22,0.17,0.22,0.21,0.37,0.17,0.33,0.28,0.38,0.28,0.32,0.38,0.32,0.22,0.19,0.23,0.01,0.97,0.02,0.91,0.01,0.86,0.01,0.89,0.02,0.88,0.9,0.03,0.02,0.02,0.11,0.02,0.01,0.03,0.88,0.01,0.06,0.87,0.07,0.86,0.01,0.33,0.39,0.26,0.23,0.23,0.29,0.34,0.29,0.38,0.18,0.19,0.28,0.32,0.25,0.32,0.22,0.23,0.12,0.17,0.23,0.28,0.29,0.24,0.32,0.33]);

trials_1_25=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25];
trials_26_50=[26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50];
trials_51_75=[51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75];
trials_76_100=[76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100];