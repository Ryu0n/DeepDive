import re
import torch
import random
import numpy as np

from prefix_gptneox_model import PrefixGPTNeoXLMHeadModel
from args import Args


def load_trained_model(model_checkpoint='weights/checkpoints/checkpoint-6720/pytorch_model.bin'):
    args = Args()
    model = PrefixGPTNeoXLMHeadModel(args)
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict=state_dict, strict=False)
    return model


regex_emoji = re.compile(':.*:')
def remove_emoji(prompt: str):
    return regex_emoji.sub('', prompt)


regex_tag = re.compile('<.*>')
def remove_tag(prompt: str):
    return regex_tag.sub('', prompt)


def generate_texts(model, prompt: str):
    model.eval()
    model.to('cuda')
    start_tokens = f'{model.tokenizer.bos_token}{prompt}'
    model_in = start_tokens + f'{model.tokenizer.eos_token}{model.tokenizer.bos_token}'
    inputs = model.tokenizer([model_in], max_length=256, return_tensors="pt", add_special_tokens=True)
    inputs.to('cuda')
    generated_ids = model.generate(inputs["input_ids"], \
            attention_mask = inputs["attention_mask"], \
            num_beams=1, min_length=32, do_sample = False, \
            max_length=128, repetition_penalty = 1.2, no_repeat_ngram_size = 3, early_stopping = True)

    result = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)[0]
    results = result.split(' [A] ')
    print('Q : ', prompt)
    for result in results[1:]:
        print('A : ', remove_tag(result).replace('[EOS]\n', ' '))
    print()


def get_ong_prompts():
    comments = [['@hi_pangsu 스롱이 넘 잘해:불::불:', '@hi_pangsu 아이고 이게 누구야 빨클러 아니신가~ :벌린_입:'],
                ['@climb_yong2 형 안본사이에 너무 잘한댜',
                '@ooooong_93 ㅋㅋㅋㅋㅋㅋㅋㅋ 오늘 쉬고있었엌ㅋㅋㅋㅋ ㅠ 아냐 형 잘하고있어! 영상 계속 잘 보구 있다! 나 안따라잡아도 많이 올라왔는걸? ㅋㅋㅋㅋ 형두! :활짝_웃다::두_손을_들고_있는_사람:'],
                ['@hobagi_darki 짜란다짜란다~~', None],
                ['@hwan_jake 많이 배웠습니다:불::불::불:', None],
                ['@20___ba 자기관리하면서 온라인 소득 벌 수 있는 제안 드리고 싶은데 잠시 소통 가능하세요?:총격전:', None],
                ['@hwan_jake 다시 올때 얘기해주세요:불::불::불:', '@hwan_jake 알겠수~~~~!!:두_손을_들고_있는_사람::두_손을_들고_있는_사람:'],
                ['@hi_pangsu 힘이 장사야:박수::박수::박수:', None],
                ['@20___ba 헬시/뷰티/바디 사업하는 이소현입니다. 메세지 가능하신가요~?:토끼풀:', None],
                ['@sssssu_jj 살살해...', '@sssssu_jj 금욜날 벽타니?'],
                ['@climb_99 와 저도 빨강파랑 런앤점프 하고 싶었는데 저기 사람 진짜 많아서 한번도 못 뛰고 왔었어요.. 부럽... 근데 뛰어도 못했을 것 같긴해요',
                '@climb_99 강해져라 동주여 할수있다! :불:'],
                ['@hhsukk_ 나도 탁탁탁…하고싶었는데:울다:', '@hhsukk_ 쑥더링 가쥬아 :벌린_입:'],
                ['@climb_yong2 첫 빨강 아주 만족:불:', '@climb_yong2 존버 해버리는거야~ :불:'],
                ['@sonyoonchocopie :박수::박수:', '@sonyoonchocopie :고양이2:\u200d:검은색_큰_정사각형::웃고_있는_고양이::고양이2:\u200d:검은색_큰_정사각형::웃는_고양이:'],
                ['@jimination_ ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 헷갈리는 탑 많지?', None],
                ['@gyoha_h 뭐냐ㅋㅋㅋㅋ 너 실력초과로 대규모 신입벙 못 가야 하는거 아니냐 ㅋㅋㅋㅋ',
                '@ooooong_93 응??? 너도 손목다침????'],
                ['@jimination_ 역시 갓 다이노~~ :박수::박수:', '@ooooong_93 ㅋㅋㅋㅋ 거생했다 형 ~'],
                ['@jungse.a 오 ㅁ쪄', None],
                ['@sssssu_jj 분하다', '@ooooong_93 머르겠어...담주도 바쁠거 같은데...:울다: 우울하구만'],
                ['@elin_ravii 릴스 돌려보는데 이게나오넼ㅋㅋㄲ', None],
                ['@jimination_ 진짜 오랜만에 봤는데 실력도 올라가고 살도 많이 빠지고 믓쪄 ~~ !! :+1::두_손을_들고_있는_사람:',
                '@ooooong_93 :두_손을_들고_있는_사람::두_손을_들고_있는_사람::+1: 응!ㅋㅋㅋㅋ 담엔 같이 강해지고 가자 ~ :기쁨:'],
                ['@jimination_ 오 여기 암장 이쁘다 ~ :벌린_입: 뛰는거 조은데?', None],
                ['@hwan_jake 대박적..! 보라색:불::불::불:', None],
                ['@k_tanglang 짱이야 형 :불:', None],
                ['@jimination_ 와 어때 오픈점', '@jimination_ 채광맛집 :두_손을_들고_있는_사람::두_손을_들고_있는_사람: 마곡 정도 넓이 되는거같어 깔끔'],
                ['@onlychu.u 1인 기업 CEO 김지수 입니다 ◡̈︎ 잠시 소통가능하세요~?:미소짓는_얼굴:', None],
                ['@climb_minimi 물파랑 추천좀,,,,,,,,,,,', '@climb_minimi 일단 초록홀드부터 혼쭐내면 될듯!! :두_손을_들고_있는_사람::두_손을_들고_있는_사람:'],
                ['@zzong_a6 와…뿌셨넹… 맨마지막 파랑 맞아…?', '@zzong_a6 저건 찐파랑...? 복수했다! 예전에 못풀었었는데 :불::불:'],
                ['@jimination_ 와 다이노 ~~ :불::불:', None],
                ['@2sso_body 자기관리하면서 온라인 소득 벌 수 있는 제안 드리고 싶은데 잠시 소통 가능하세요?:총격전:', None],
                ['@jimination_ 강려크해 ~~ :불::불:', None],
                ['@climb_minimi 이야 내가 말해준거 다 뿌셨네 :박수::박수:',
                '@climb_minimi 다음 정보 업데이트 부탁드립니다~ 꿀남색 너무 달잖어 :두_손을_들고_있는_사람::두_손을_들고_있는_사람:'],
                ['@jimination_ 형 살 많이 빠졌다.. :벌린_입:', '@ooooong_93 진짜?! 대박.. :불::불::불:'],
                ['@jimination_ 와 빨강..!!! :불::불::불: 와 형 안본사이에 무브 진짜 조아졌다 ~~ :+1::+1:',
                '@ooooong_93 형도 코디랑 다이노해벜ㅋㅋㅋㅋ 꿀빨강을 못찾겠어 ㅠ'],
                ['@sssssu_jj 머여 오늘 간건 왜 안올린겨', '@sssssu_jj 오늘간건 다음 주간에 ㅋㅋㅋ:박수::박수:'],
                ['@jimination_ 실력이 쭉쭉 올라가 ~~ :불::불: 미역줄깈ㅋㅋㅋㅋ', '@ooooong_93 ㅋㅋㅋㅋ 기다리고 있을게!'],
                ['@sj.from_0706 삼각대!', None],
                ['@kunyo_os 아 첫번째 영상 편집영상인줄 알고 언제성공하나 계속봤네 ㅋㅋㅋㅋㅋ:기쁨:', None],
                ['@jimination_ 빨강클라이머 가즈아! :불:', '@ooooong_93 뛰뛰도 함가야지 :불:'],
                ['@gyoha_h 축하해여:)', '@gyoha_h 교마워~~ :두_손을_들고_있는_사람:'],
                ['@zz.hi.yo 지려잇!!', None],
                ['@hwan_jake 굿입니다!', None],
                ['@jungse.a 손가락 노련했다 ㅋㅋㅋㅋㅋ', None],
                ['@jimination_ 신림 더클 어때??', None],
                ['@sssssu_jj ㅋㅋㅋㅋㅋㅋㅋㅋ괜찮아 실력으로 극복했어!!', None],
                ['@zz.hi.yo 키야 지리네요:박수:', None],
                ['@gyoha_h 와... 실력 엄청 오르셨네요 ㅎㅎㅎ', '@gyoha_h 파랑 길좀 봐주어어어ㅓㅓ:두_손을_들고_있는_사람::두_손을_들고_있는_사람:'],
                ['@sonyoonchocopie :+1::피부톤-2::+1::피부톤-2::+1::피부톤-2:', None],
                ['@gucci0419 나도 데꼬가~~:불::불:', '@gucci0419 여기도 좋더라구여~ 암장 컬렉션 +1'],
                ['@jungse.a 도장깨기??!:불::불::불:', '@jungse.a 언제나 깨지는건 나:울음:'],
                ['@k_tanglang 옹형 담주에도 도저언:불:', '@k_tanglang 담엔 같이 시도해보자!'],
                ['@sssssu_jj 담주에 한 번 더 하면 이제 잡을 수 있어', '@sssssu_jj 이거부터 처리해버리고 다른거해야겠네 ㅋㅋㅋ'],
                ['@jungse.a 오빠 이제 클아일체네.. 중력 방향 바뀐듯',
                '@ooooong_93 원데이클래스?? 일일강습? 할지 아예 좀더 중장기로 배우는거 있던데 그거 할지 고민']]
    prompts = [comment[0] for comment in comments]
    return prompts


if __name__ == "__main__":
    # prompts = [
    #     '나 지금 너무 심심한데 랩이나 한곡 해봐.',
    #     '피곤해 안피곤해지는 방법 뭐가 있지?',
    #     '디자인을 잘하는 방법은 무엇일까?',
    #     '날씬한 사람들의 이유는 무엇일까?',
    # ]
    prompts = get_ong_prompts()
    model = load_trained_model()
    for prompt in prompts:
        prompt = remove_emoji(prompt)
        generate_texts(model, prompt)
