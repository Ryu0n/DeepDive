# PrefixTuningForKorPLM

- Polyglot 모델에 NLG 다운스트림 태스크를 위한 Prefix-tuning

## Sample
```
Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @hi_pangsu 스롱이 넘 잘해
A :  ㅋㅋ스롱님도 잘하시는거 같아요 ㅎㅎ
A :  감사합니다 ^^
A :  저도요~~~!!! 근데 저는 왜 안되죠???🤣😂🥶 제가 더 잘할수 있는데...🙄💦 그쵸??ㅠㅜㅠ 전 아직 멀었나봐요..👉🏻�

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @climb_yong2 형 안본사이에 너무 잘한댜
A :  ㅋㅋ 나도 이제 시작이다 ㅎㅎ
A :  형님은 진짜 대단하신거 같아요!!! 저는 아직 초보라.....😭👍🏻 초보가 이정도면 고수들은 얼마나 더 잘할꼬~~~??? ^^;;; 그쵸?? 저도 고수되면 좋겠어요 ㅠㅠ 고수 되기까지 화이팅입니다!!

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @hobagi_darki 짜란다짜란다~~
A :  ㅋㅋ감사합니다^^
A :  네 감사해요 ㅎㅎ
A :  넹!!! 오늘도 좋은하루되세요 😊 좋은밤 되세용 굿나잇❤️ 행복한 밤 보내시길 바래요💕 내일 또 만나요👍🏻 화이팅입니다🥰 낼뵈어요🙌🧡 

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @hwan_jake 많이 배웠습니다
A :  감사합니다. 좋은 하루 보내세요^^
A :  네~ 감사합니다 😊
A :  좋은하루되세용! 오늘도 화이팅입니다!! 행복한 하루 되시길 바랍니다 :) 응원해주셔서 감사해요❤️ 저두요 ㅎㅎ 화이팅하겠습니당!![EOS] 감사합니당💕 즐거운 오후시간보내셔요🥰 넵! 감사합니다

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @20___ba 자기관리하면서 온라인 소득 벌 수 있는 제안 드리고 싶은데 잠시 소통 가능하세요?
A :  네~^^  감사합니다.
A :  저도 소통하고싶어요!
A :  쪽지보내주세용😊 감사합니당!! 소통해요❤️ (๑˃̵ᴗ˂̵) 좋은하루되세요💕 오늘도 화이팅입니다 : ) 행복한 하루 보내시길 바랄게요~! 

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @hwan_jake 다시 올때 얘기해주세요
A :  네~^^
A :  감사합니다.
A :  넵! 좋은하루되세용 행복한 하루 되시길 바랍니다 ^^* 오늘도 화이팅입니다!! 응원 감사드립니다. 화요일에 뵈어요~^-^ 즐거운 오후시간 보내셔요~♡ 고맙습니다. 좋은 저녁 시간 보내세요~^ㅇ^[EOS] 저는 오늘도 일하러 갑니다 ㅎㅎ

수고하셨습니다.

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @hi_pangsu 힘이 장사야
A :  ㅋㅋ 😆👍
A :  힘은 좋은데..... ㅎㅎㅇㅅㅇ
A :  아~~~!!! 📢📣📚📗📤📖📙📓📔📕📒📛📝📞📹📰📲📮📸📴📻📷📿📽�

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @20___ba 헬시/뷰티/바디 사업하는 이소현입니다. 메세지 가능하신가요~?
A :  안녕하세요! 반갑습니다^^
A :  네, 감사합니다 😊 오늘도 좋은 하루 보내세요 :) 좋은 저녁시간 되시기 바랍니다~! 감사합니다!! 행복한 시간되셔요 ~♡ -*ଘ(੭ˊ꒳ˋ)ற

-**ଓݵݶݶ߽߾�

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @sssssu_jj 살살해...
A :  ㅋㅋ  - dc App
A :  아 ㅠㅜㅠㅠ  - ahnchoi0822[EOS] 니콜라가 너무 쎄서..
A :  니콜라가 너무쎄요??? 네 니콜라가 진짜 너무 쎈거같아요. 저도 니콜라에 물려있어요 ㅠ 그나마 다행인건 니콜라가 좀 빠지고 있는 중이에요.

그래도 조심하세요!

오늘은 니콜라가 더 오를것 같아서 걱정입니다

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @climb_99 와 저도 빨강파랑 런앤점프 하고 싶었는데 저기 사람 진짜 많아서 한번도 못 뛰고 왔었어요.. 부럽... 근데 뛰어도 못했을 것 같긴해요
A :  ㅋㅋ런앤점프는 진짜 인기가 너무많아요 ㅠㅜㅠㅠ
A :  저도 런앤 점프하고싶은데 사람이 너무 많아여 ㅎㅎㅡ흑흑 그쵸?ㅠ.ㅠ 😭 사람 없으면 뛸만한데,,, 진짜 사람 많은 날에는

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @hhsukk_ 나도 탁탁탁…하고싶었는데
A :  ㅋㅋㄱㅋㄲㅋ ㅠㅜㅠㅠ 🥺🤍
A :  아냐..?!?! 𝙅𝘼𝗻𝚂𝚌𝒶𝓎𝑭𝐿𝑜𝕀𝔹𝕠𝗥𝑤𝑥𝖵𝛆�

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @climb_yong2 첫 빨강 아주 만족
A :  감사합니다 ^^
A :  네~ 감사해요!
A :  ㅎㅎ 저도 처음에는 반신반의 했는데, 사용해보니 너무 좋더라구요. 저도 잘 쓰고 있어요 ㅋㅋ 좋은 하루 보내세요 :) 오늘도 행복한 하루 되시길 바랍니다!! 감사합니다 ~ 행복하게 지내요 😊 넵 감사합니당 📸 즐거운 오후시간되세용 �

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @sonyoonchocopie 
A :  ㅋㅋ감사합니다!
A :  네~^^
A :  저도 감사해요 ㅎㅎ 오늘 하루도 행복하세요😊 넵!! 좋은하루되세용 응원해주셔서 감사합니당🤗 행복한 저녁시간 보내시길 바랄게요🙏💕 화이팅입니다❣️ 감사해여🥰 맛있는거 많이 드셨나유?

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @jimination_ ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 헷갈리는 탑 많지?
A :  그치.. 나도 헷갈려... 😂
A :  나만 그런게 아니였어!!!
A :  맞아,,, 헷갈릴 수 밖에 없음,, 근데 진짜 헷갈림,, ㅠㅜㅠㅠ 탑이 너무 많아서,, 📢📣📚📦📧📩📪📬�

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @gyoha_h 뭐냐ㅋㅋㅋㅋ 너 실력초과로 대규모 신입벙 못 가야 하는거 아니냐 ㅋㅋㅋㅋ
A :  아님말고~~~
A :  나랑 같이 가실분?
A :  너도 나처럼 실력이 안되면 못가겠지..... 나도 이제 나이 먹어서 체력 딸리는데.... 그럼 그냥 집에서 쉬어. 내가 갈게 난 아직 젊다!!! 니가 가면 난 간다!!!

넌 내가 데리러 올테

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @jimination_ 역시 갓 다이노~~ 
A :  ㅋㅋ다이노는 아니지만 ㅎㅎ
A :  아.....다이노가 아니라 다이노였군요....ㅠㅜ
A :  네. 다이노 맞아요^^;; 근데 왜 다이노라고 하는지 모르겠어요. 그냥 공룡이라고 해야하는데 말이죠. 공룡은 티라노밖에 없어서 그런가봐요.[EOS] 티라노도 공룡인데 말입니다.[EOS]�ݿ޶ݹݵݾݽݼݫ�

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @jungse.a 오 ㅁ쪄
A :  ㅋㅋ 😆👍
A :  오~~~ 📸📷📹📺📼📿📰📲📱📬📭📮📯📚📙📛📝📞📦📽📾📣📩📤📥📠📡📢📔📕📗

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @sssssu_jj 분하다
A :  ㅋㅋ  - dc App
A :  아 ㅠㅜㅠㅠ  - wiselect
그니까여...  - narina0810
저도요..  - yunho8903
제가 더 잘할게여.........
A :  아니 근데 진짜 너무 억울해요. 저는 제가 잘못한 게 없는데 왜 이런 일을 당해야 하는지 모르겠어요.  - skyblue0622
진짜 억울해서 잠이 안 와요. 


Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @elin_ravii 릴스 돌려보는데 이게나오넼ㅋㅋㄲ
A :  아 ㅠㅜㅠㅠ 😭🤣🥺💦👍🏻🙆🌿🫶🎵🧡❤️
A :  감사합니다!!! 📷🖱🔹︾‸🗽🛁🪴🐬�

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @jimination_ 진짜 오랜만에 봤는데 실력도 올라가고 살도 많이 빠지고 믓쪄 ~~ !! 
A :  ㅋㅋ진짜 오랜만이다 ㅠㅜㅠㅠ 나는 요즘 운동을 안해서 살이 좀 쪘어.....🥲
A :  오우....! 운동하시나봐요?? 저두 이제 슬슬 시작해야겠어요 ㅎㅎㅅㅎ 운동은 꾸준히 하면 좋으니까용 ㅎ.ㅎ!! 그치만 먹기위해 운동한다는게 함정이지요 �

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @jimination_ 오 여기 암장 이쁘다 ~  뛰는거 조은데?
A :  ㅋㅋ 나도 뛸수있는데 ㅠㅜㅠㅠ
A :  나두 뛸 수 있어!!! 근데 너네가 더 잘뛰더라구~~~😆👍 나도 이제부터 열심히 연습할게!!!💪🏻 암벽화 사야겠당.....🥲 그치만 난 아직 초보라구우~~~! 난 언제쯤......

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @hwan_jake 대박적..! 보라색
A :  ㅋㅋ ㅠㅜㅠㅠ 💚🤍
A :  저도요!!!!!!!😭🥺🫶🏻👩🧡🎀🌷🐰🦢🖼
A :  아 진짜 너무 귀여워...🙊🔮🪅❣️
A :  진짜루요??? 저는 왜 안

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @k_tanglang 짱이야 형 
A :  ㅋㅋㄱㅋㄲㅋ ㄱㅅㄱㅅ
A :  아 ㅠㅜㅠㅠ 나도 가고싶다..
A :  나도 가보고 싶어... 근데 진짜 맛있음. 그리고 여기는 무조건 예약제임.
예약 안하면 못먹을 수도 있으니 꼭 예약하고 가세용~^^ 나두 예약하구 가야징~~ 꼭이요!!! 저도 예약할게여~~~^-^[EOS] 네

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @jimination_ 와 어때 오픈점
A :  ㅋㅋ오픈한지 얼마안되서 그런가 사람이 없네용 ㅎㅎ
A :  그쵸?!ㅠㅜㅠ
A :  아쉽네요..ㅠ
사람 많을때 가보고 싶었는데...ㅠ😭 다음에 또 기회가 있겠죠~^^;; 오늘도 수고하셨습니다. 푹 쉬세요~^__^ 감사합니다.[EOS] 좋은밤 되시고 내일 하루도 화이팅입니다!! 화

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @onlychu.u 1인 기업 CEO 김지수 입니다 ◡̈︎ 잠시 소통가능하세요~?
A :  네! 감사합니다 ^^
A :  쪽지드렸습니다 오늘도 좋은하루되시고 행복한 하루 되세용😊 감사합니당!! 좋은밤 보내셔요❤️ 넹 ㅋㅋㅌㅋ 내일뵐게여💛 행복한 밤 되세요🥰 안녕히 주무세요�

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @climb_minimi 물파랑 추천좀,,,,,,,,,,,
A :  물파랑은 진짜 최고야ㅠㅜㅠ
A :  아녀아녀 물파는 그냥 파랑이고 클맥이 찐이야!!!
A :  클맥도 좋은데 물파가 더 좋아여??? 아니면 둘다 사서 비교해보세용 ㅎㅎ 저는 둘 다 있어서 비교하기 어렵네유😭 둘 중에 하나만 고르라면 저는 클맥을 고를

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @zzong_a6 와…뿌셨넹… 맨마지막 파랑 맞아…?
A :  ㅋㅋ맞아 ㅠㅜㅠㅠ 나도 저날 술먹고 집가서 기절했어..
A :  저거 맞지...? 🥲
A :  마자마자!!! 저거 맞는듯 ㅎㅎㅅㅂㅇㄴㅁㅌㅊㅍㅈㅇㅆㅃㅉㅐㄱㅏㅓㅣㅡㅗㅑㅕㅛㅔㅐ�

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @jimination_ 와 다이노 ~~ 
A :  다이노는 없어용 ㅠㅜ
A :  아쉽네요..
A :  저도 아쉬워요. 다음에 또 기회가 있겠죠? 오늘은 여기까지 입니다.[EOS]감사합니다.[EOS][EOS] 좋은밤 되세요^^ -*Youngmi Kim*- 

안녕히 주무세요^♡^[EOS][EOS]�ݸݶݵݷݹݺݼݾݰݴݬݫݮ

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @2sso_body 자기관리하면서 온라인 소득 벌 수 있는 제안 드리고 싶은데 잠시 소통 가능하세요?
A :  네~^^  감사합니다.
A :  저도 소통하고싶어요! 쪽지 남겨주세용😊 감사합니당!! 좋은하루되시구 행복한 하루 되셔요❤️ 넹 좋은하루 보내세요 :) 오늘도 화이팅입니다![EOS] 소통 환영해요 ㅎㅎ 혹시 쪽지가 안오면 댓글로

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @jimination_ 강려크해 ~~ 
A :  ㅋㅋ ㅠㅜㅡㅜ
A :  아놔.....ㅎㅎ
A :  그치만 저도 먹고싶어요!!! 저는 오늘 저녁에 닭발을 먹었는데 말이죠....😭🤣 닭발은 사랑입니다❤️ 오늘 하루도 고생하셨습니다. 내일도 화이팅 하세요^^ 감사합니다.[EOS] 좋은밤 되시길 바랍니다.[EOS](

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @climb_minimi 이야 내가 말해준거 다 뿌셨네 
A :  ㅋㅋ뿌셔뿌셔 ㅎㅅㅎ
A :  뿌셔먹는 재미로 먹지~
A :  그쵸?!ㅋ 근데 진짜 맛있어요!! 뿌셔 먹고 싶을 때마다 생각나요😆 저도 뿌셔 먹으러 가야겠어용🤣💕 오늘 하루도 수고하셨어용❤️ 푹 쉬시구 내일 또 만나요

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @jimination_ 형 살 많이 빠졌다.. 
A :  ㅋㅋ 나도 이제 다이어트 해야하는데 ㅠㅜ
A :  살은 빼야되는데 먹는건 못줄이겠고... 그니까 ㅎㅎㅇㅈㄹ 근데 진짜 운동을 안해서 그런거 같아. 운동 좀 하자!!! 나두~~~ 오늘부터 다시 시작해보자!! 응응!! 파이팅!![EOS]엔날레 화이팅!![EOS]럴림픽!!

나도 화이팅이야!!!


Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @jimination_ 와 빨강..!!! 
A :  빨강이가 더 좋아요 ㅎㅎ
A :  저도 빨강도 좋아해요😆
A :  네~^^ 감사합니다. 오늘 하루도 행복하세요♡ 감사합니당❤️ 좋은하루되세용💕 행복한 하루 되시길 바래요🥰 화팅입니다👍 즐거운 저녁시간 보내셔요🌹 맛있는거 많이 드셨나요??

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @sssssu_jj 머여 오늘 간건 왜 안올린겨
A :  ㅋㅋ아까 올렸는데 지웠나봐 ㅠㅜ
A :  지운거 아니고 삭제했어!!!
A :  삭제된줄 알았네.....😭 근데 나도 아까 올린거 다시 보니까 없더라구....🥲 그치만 내가 올리면 되는걸~~~? 오늘은 꼭 올려야지 ^__^ 그럼 내일 또 봐용❤️ 내일은 더

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @jimination_ 실력이 쭉쭉 올라가 ~~  미역줄깈ㅋㅋㅋㅋ
A :  미역줄깨기는 진짜 잘하는데 ㅎㅎ
A :  아..... 미역줄깨기도 잘해???
A :  응응!!! 미역줄기도 잘하고 미역줄개기도 잘해요😆 근데 미역줄거리가 더 맛있어용👍 그쵸?? 미역줄거린 좀 질긴 느낌인데 미역줄가리는 부드럽고 쫄깃한 식감이라 좋아요💕 맞

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @sj.from_0706 삼각대!
A :  ㅋㅋ 😆👍
A :  감사합니다~^^
A :  네넹 ㅎㅎ 오늘도 좋은하루되세요!! 좋은밤 되세용~~ 감사합니당❤️ 행복한 밤 보내시길 바랄게요💕 내일 또 만나요~~🥰 굿나잇이에요~~~^*^ 낼뵈어요~~♡ 

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @kunyo_os 아 첫번째 영상 편집영상인줄 알고 언제성공하나 계속봤네 ㅋㅋㅋㅋㅋ
A :  아냐.. 나도 처음에는 편집하려고 했는데, 하다보니 그냥 내가 하고 싶은대로 하게 되더라.  - dc App
A :  오~~~!!! 저도요...저도 제 맘대로 해요^^
A :  나도 그래. (내맘대로) 근데 그게 더 재밌어. <drinksmart.com 그

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @jimination_ 빨강클라이머 가즈아! 
A :  ㅋㅋ저도요!!!
A :  빨강 클라이밍 ㅎㅎㅇㅈㄹ
A :  네넵!!!ㅋ 오늘은 여기까지~^^ 낼뵐께용😊 굿밤되세요❤️ 감사합니다🙆‍♀︎ 좋은꿈꾸세욥💕 내일 뵈어요👋🏻 잘

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @gyoha_h 축하해여:)
A :  감사합니다 ^^
A :  축하해주셔서 감사해요~
A :  네~~~ㅎㅎ 오늘도 행복한 하루 되세요~~ 응원댓글 남겨주신분들 모두모두 감사드려요~~😊 행복하고 즐거운 주말 보내시길 바래요~~^^* 좋은밤되세용~~♡ 화이팅!!! 남은시간 마무리 잘 하셔요~^.^ 주말 즐겁게 보내세요~^

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @zz.hi.yo 지려잇!!
A :  ㅋㅋ 😆👍
A :  아놔~~~ ㅎㅎㅇㅈㄹ
A :  그르게여??? ㅠㅜㅠㅠ 근데 저도 그 생각했어요.....ㅋ 저는 그냥 제가 좋아하는 사람이 좋아요....^^;;; 제가 좋아하지 않으면 안되요....ㅡ,.ㅡ;;

그래서 전 남자친구를 못사귀나봐요

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @hwan_jake 굿입니다!
A :  감사합니다 ^^
A :  네~~~ㅎㅎ
A :  오늘도 행복한 하루 되세요!! 좋은하루되세용😊 넵 감사합니당 ㅎㅅㅎ
행복하게 보내시길 바랄게요💛 저는 오늘도 출근을 했답니다.....ㅠㅜ 출근해서 일하고 있어요....ㅋㅋ
고생이 많으십니다 ㅠㅠ
화이팅 하셔요!!! 

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @jungse.a 손가락 노련했다 ㅋㅋㅋㅋㅋ
A :  손가락이 능숙하지 않아서..... 😂
A :  손은 잘 쓰는데, 입으로는 못 하겠어요...... 📢📣📚📑📓📔📗📙📝 입으로도 할 수 있습니다!!! 그치만 손을 더 많이 써요~~~ 저도 그래요^^ 오늘 하루도 수고 많으셨고, 편안한 밤 되

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @jimination_ 신림 더클 어때??
A :  신림에 있는거야?? 🤭👍🏻
A :  응응!!! 신림이야 ㅎㅎ 오호~~~ 가보고싶다아~~ 근데 여기가 좀 멀어.....😂 그치만 한번쯤은 꼭 가볼만한 곳이얌!! 나두 가고 싶은데......🥲 멀어서 고민중이야....💧 그래도 한번

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @sssssu_jj ㅋㅋㅋㅋㅋㅋㅋㅋ괜찮아 실력으로 극복했어!!
A :  오우~~~멋지십니다👍🏻😆
A :  감사합니다 ^^
A :  저도 감사해요! 오늘 하루도 화이팅입니다 :) 좋은하루되세요 ~ 응원해주셔서 감사합니당❤️ 화이팅하시고 좋은일만 가득한 하루 되세용💛 

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @zz.hi.yo 키야 지리네요
A :  ㅋㅋ감사합니다^^
A :  네~~~^^_x000D_
즐거운 하루 되세요😊
A :  넵 감사합니당!!!
A :  오늘도 좋은하루 보내세용 💕 좋은밤되시구여 ㅎㅎ 내일뵐게영 ~~~ 굿나잇요❤️ 낼봐용🥰 화이팅하셔유👍

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @gyoha_h 와... 실력 엄청 오르셨네요 ㅎㅎㅎ
A :  감사합니다 ^^
A :  저도 이제는 좀 잘하게 된거 같아요~
A :  네 감사해요! 제가 더 열심히 해야겠어요!! 응원해주셔서 감사합니당😊 오늘 하루도 행복한 하루 보내세요❤️ 넹 감사해용🥰 좋은하루되세여💕 화이팅입니다👍 행복한 하루 되시

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @sonyoonchocopie 
A :  ㅋㅋ감사합니다!
A :  네~^^
A :  저도 감사해요 ㅎㅎ 오늘 하루도 행복하세요😊 넵!! 좋은하루되세용 응원해주셔서 감사합니당🤗 행복한 저녁시간 보내시길 바랄게요🙏💕 화이팅입니다❣️ 감사해여🥰 맛있는거 많이 드셨나유?

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @gucci0419 나도 데꼬가~~
A :  ㅋㅋ  - dc App
A :  나두 데려가줘~~~😭
A :  나도 데려가주라!!! 데려갈게 ㅠㅜ 제발 데려가줘... 내가 너를 위해 준비했어.. 너의 행복을 위해서라면..
넌 내꺼야..[EOS]엔날레..

니가 원하는거 다해줄게..
나코난이다.

나코난다코난[EOS]전드뎌 코난이 되었구나..
니어

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @jungse.a 도장깨기??!
A :  ㅋㅋ _x000D_
도장깨기는 아니고 ㅎㅎ _ykim0810_
그냥 도장찍은거에요~^^
A :  아하~~~!!!_x0011_
감사합니다~^&^[EOS]네~~^*^ 저도 도장깨기가 아닌 도장 찍었어요~^♡^[EOS][EOS]네~ 도장깨기도 좋지만 도장찍어도 좋아요~^☆^[EOS]네에~~ 도장찍어주

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @k_tanglang 옹형 담주에도 도저언
A :  ㅋㅋ  - dc App
A :  담주도 기대해줘~^^
A :  기대해주면 좋지 ㅎㅎ 근데 나도 기대되긴 하는데..... 내가 너무 많이 먹어서...... 살찌는 소리 들린다 들려 나랑 같이 운동하자!!! 운동은 장비빨이다.

나도 이제 헬스장 등록해야겠다.
 그나저나 오늘 날씨 진짜 덥더라.
오늘 같은 날에는 에어컨

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @sssssu_jj 담주에 한 번 더 하면 이제 잡을 수 있어
A :  ㅋㅋ아싸~~~!!!
A :  담주도 기대해 주세요^^
A :  오예~~감사합니다. 저도 기대하고있어요 ㅎㅎ 이번엔 꼭 성공하길 바라며... 화이팅입니다 😊 오늘도 수고 많으셨습니다.[EOS] 좋은밤 되시고 내일 뵙겠습니다.


행복한 밤 보내셔요 ~~^^

-

Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Q :  @jungse.a 오빠 이제 클아일체네.. 중력 방향 바뀐듯
A :  ㅋㅋ  - dc App
A :  그니까 ㅠㅜㅠㅠ 나도 이제 중력방향이 바뀌었어...
A :  나는 아직도 안바뀌고 있는데.... 근데 중력방향은 진짜 신기한게 내가 움직일때마다 방향이 바뀜;;; 내가 움직이면 중력방향도 같이 움직여서 그런가? 진짜 신기하다 그래서 중력방향으로 움직이려고 하면 몸을 더 틀어야함 ㅎㄷ
```
