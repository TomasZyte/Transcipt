"""Build a private, offline review release from existing WhisperX evidence.

No env file, model loading, network or cloud settings. Output must not exist.
The result is a human-review artifact, never a deployment package.
"""
import argparse
from array import array
import html
import json
from pathlib import Path
import shutil
import sys
import wave

import saved_benchmark_review as benchmark
from asr_comparison import digest, read_json
import channel_review
import channel_speech
import channel_asr
import channel_pipeline


def deny_network(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo', 'socket.bind', 'subprocess.Popen', 'os.system'):
        raise RuntimeError('Offline release forbids networking and child processes')


def save(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


STYLE = '''body{font:17px/1.55 system-ui;background:#f3f5f8;color:#182338;max-width:1160px;margin:28px auto;padding:0 20px}
section,article{background:white;border-radius:10px;padding:18px;margin:14px 0}article.flag{border-left:5px solid #ba7600}
article.missing{border-left:5px solid #a82836}aside{background:#fff0ca;padding:18px;border-radius:8px}
audio{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px}summary,button{cursor:pointer}
button{font:inherit;padding:6px 12px}small{color:#465368}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:8px;text-align:left;border-bottom:1px solid #ccd2da;vertical-align:top}.player{position:sticky;top:0;z-index:2}
label{display:inline-block;margin:8px}a{color:#154f98}h2{font-size:22px}.hidden{display:none}'''


def document(title, body, script=''):
    return '<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'+\
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; media-src \'self\'; style-src \'unsafe-inline\'; script-src \'unsafe-inline\'">'+\
        '<title>'+html.escape(title)+'</title><style>'+STYLE+'</style><h1>'+html.escape(title)+'</h1>'+body+\
        ('<script>'+script+'</script>' if script else '')+'</html>'


def phone_html(review):
    esc = html.escape
    body = '<p><a href="index.html">К результатам проверки</a> · <a href="call-draft.txt">Скачать TXT</a></p>'
    body += '<aside><b>Черновик для проверки.</b> Все сохранённые окна речи, без повторного распознавания. '
    body += 'Ни одна строка не считается подтверждённой по звуку. Канал 1 — левый, канал 2 — правый; роли и постоянство говорящего на всём звонке не проверены. '
    body += 'Разногласие двух проходов — повод проверить, а не доказательство ошибки. Пропуски VAD возможны.</aside>'
    body += '<section class="player"><audio id="player" controls preload="metadata" src="media/call-stereo.wav"></audio>'
    body += '<p>Время: <output id="clock">0.000</output> с. <button onclick="player.playbackRate=.75">0,75×</button> <button onclick="player.playbackRate=1">1×</button></p>'
    body += '<label><input id="only" type="checkbox"> Только помеченные участки</label></section>'
    body += '<p>Помечено: '+str(review['counts']['requires_review'])+' из '+str(review['counts']['speech_runs'])+' участков. '
    body += 'Интервалы строк — исходные окна речи; для сомнительных слов это не точный тайминг.</p>'
    for row in review['rows']:
        flagged = bool(row['reasons']); missing = row['text'] is None
        cls = 'missing' if missing else 'flag' if flagged else ''
        text = row['text'] if not missing else '[НЕРАЗБОРЧИВАЯ РЕЧЬ — текст не восстановлен]'
        body += f'<article class="{cls}" data-flag="{int(flagged)}" id="{row["id"]}">'
        body += f'<button onclick="playRange({row["startSec"]},{row["endSec"]})">▶ {channel_review.timestamp(row["startSec"])}–{channel_review.timestamp(row["endSec"])}</button> '
        body += '<b>Канал '+('1' if row['channel']=='left' else '2')+'</b>'
        body += '<p>'+esc(text)+'</p>'
        if flagged:
            body += '<small>ПРОВЕРИТЬ: '+esc('; '.join(channel_review.REASONS.get(r,r) for r in row['reasons']))+'</small>'
        if row['overlaps']:
            body += '<p><small>Перекрытие с другим каналом: '+', '.join(f'<a href="#{i}">{i}</a>' for i in row['overlaps'])+'</small></p>'
        body += '<details><summary>Исходная гипотеза и времена слов</summary><p>'+esc(row['original_text'])+'</p><pre>'+esc(json.dumps(row['words'],ensure_ascii=False,indent=2))+'</pre></details></article>'
    body += '<section><h2>Контрольное распознавание полных каналов</h2><p>Сохранённый прежний проход; его текст и времена также могут быть ошибочны.</p>'
    for channel, text in review['control_text'].items():
        body += '<details><summary>'+esc(channel)+'</summary><p>'+esc(text)+'</p></details>'
    body += '</section>'
    script = '''const player=document.getElementById('player'),clock=document.getElementById('clock');let stopAt=null;
function playRange(a,b){player.pause();player.currentTime=Math.max(0,a-.25);stopAt=Math.min(player.duration||b,b+.25);player.play();}
player.addEventListener('timeupdate',()=>{clock.textContent=player.currentTime.toFixed(3);if(stopAt!==null&&player.currentTime>=stopAt){player.pause();stopAt=null;}});
document.getElementById('only').addEventListener('change',e=>document.querySelectorAll('article').forEach(a=>a.classList.toggle('hidden',e.target.checked&&a.dataset.flag==='0')));'''
    return document('Звонок: полный черновик с пометками', body, script)


def benchmark_html(report):
    esc = lambda x: html.escape(str(x))
    body = '<p><a href="index.html">К результатам проверки</a></p><aside>Повторная сборка сохранённых данных без моделей и сети. '
    body += 'Техническая воспроизводимость не означает правильность слов или голосов. Число меток не подтверждает число людей. WER/DER не измерялись.</aside>'
    for sample, note in zip(benchmark.IDS, ('Пение под музыку; минимум 3, возможно 4 голоса.',
            'Четыре человека; отделение детей от взрослых не принято.', 'Один человек.', 'Один человек.')):
        body += f'<section><h2>{sample}</h2><p>{note}</p><audio controls preload="none" src="media/{sample}.wav"></audio>'
        body += '<table><tr><th>Сохранённый вариант</th><th>Повторная сборка</th><th>Слов</th><th>Метки дорожки / текста</th><th>Слов без голоса</th><th>Слов >2 с</th><th>Детские фразы 1 / 2</th></tr>'
        rows = [r for r in report['rows'] if r['sample']==sample]
        for r in rows:
            values = [r['profile']+' / '+r['mode'], 'OK' if r['replay_pass'] else 'ОШИБКА',
                r.get('lexical_word_count','—'), str(r.get('timeline_labels') if r.get('timeline_labels') is not None else '—')+' / '+str(r.get('export_labels','—')),
                r.get('unknown_words','—'), r.get('long_words','—'), ' / '.join('есть' if x else 'нет' for x in r.get('phrase_present',[])) if sample=='sound2' else '—']
            body += '<tr>'+''.join('<td>'+esc(v)+'</td>' for v in values)+'</tr>'
        body += '</table>'
        for r in rows:
            body += '<details><summary>'+esc(r['profile']+' / '+r['mode'])+'</summary><pre>'+esc(json.dumps(r,ensure_ascii=False,indent=2))+'</pre></details>'
        body += '</section>'
    body += '<section><h2>Проверка детских реплик по сохранённой разметке владельца</h2><p>Метки взрослых привязаны к SHA каждого результата; не переносятся между запусками. Времена владельца приблизительные. Эталон использован только для оценки.</p>'
    for profile, acceptance in report['acceptance'].items():
        body += '<details><summary>'+esc(profile)+'</summary><pre>'+esc(json.dumps([r for r in acceptance['rows'] if r['sample']=='sound2'],ensure_ascii=False,indent=2))+'</pre></details>'
    body += '</section><section><h2>Технические проверки</h2><p>В ранних baseline/community нет отдельного ASR и полной дорожки: проверен только экспорт сохранённого выравнивания. Неопределённая метка в тексте не означает ещё одного человека. Ошибки контекстного эксперимента сохраняются; такие варианты исключены из кандидата.</p><pre>'+esc(json.dumps(report['checks'],ensure_ascii=False,indent=2))+'</pre></section>'
    return document('Сохранённые sound1–4: проверка регрессий',body)


def build(quality, runtime, output):
    quality, runtime, output = map(lambda p: Path(p).resolve(), (quality, runtime, output))
    if output.exists():
        raise ValueError('Never overwrite an existing review release')
    phone = quality/'phone-call-2026-10-30'
    paths = [runtime/'channel-speech-v1/jobs/result-channel-speech-v1.json',
             runtime/'channel-asr-v1/jobs/result-channel-asr-v1.json']
    states = [read_json(p) for p in paths]
    for state, module, name in zip(states, (channel_speech, channel_asr), ('channel-speech-input.json','channel-asr-input.json')):
        channel_pipeline.validate_result(state, module.prepare(phone/name))
    review = channel_review.build(*states)
    report = benchmark.review(quality, runtime)
    hashes = {**report['input_sha256'], **{str(p):digest(p) for p in paths}}
    for name in ('channel-speech-input.json','channel-asr-input.json','left-full.wav','right-full.wav'):
        hashes[str(phone/name)] = digest(phone/name)
    owner = quality/'channel-repair-2026-10-31/owner-review.json'
    notes = read_json(owner); hashes[str(owner)] = digest(owner)
    output.mkdir(); (output/'media').mkdir()
    save(output/'call-review.json', review); save(output/'benchmark-review.json', report)
    save(output/'owner-notes.json', {'purpose':'evaluation_only_not_inference', 'source':notes,
          'note':'Owner notes have not replaced any ASR words or timing in the draft.'})
    (output/'call-draft.txt').write_text(channel_review.text_export(review),encoding='utf-8')
    (output/'call.html').write_text(phone_html(review),encoding='utf-8')
    (output/'benchmarks.html').write_text(benchmark_html(report),encoding='utf-8')
    channels=[]
    for name in ('left','right'):
        with wave.open(str(phone/(name+'-full.wav')), 'rb') as wav:
            channels.append(array('h', wav.readframes(wav.getnframes())))
    stereo=array('h',[0])*(2*len(channels[0])); stereo[::2]=channels[0]; stereo[1::2]=channels[1]
    with wave.open(str(output/'media/call-stereo.wav'),'wb') as wav:
        wav.setnchannels(2); wav.setsampwidth(2); wav.setframerate(16000); wav.writeframes(stereo.tobytes())
    for name in benchmark.IDS:
        shutil.copyfile(quality/(name+'.wav'),output/'media'/(name+'.wav'))
    if any(digest(path)!=sha for path,sha in hashes.items()):
        raise ValueError('Input evidence changed')
    summary={'release':'poslovnica-review-2026-11-01','status':'prepared_for_manual_testing',
        'production_ready':False,'paid_jobs_submitted':0,'network_calls':0,'model':'WhisperX large-v3 unchanged',
        'call':review['counts'],'benchmark_output_count':len(report['rows']),
        'benchmark_replay_pass':report['technical_replay_pass'],'input_sha256':hashes}
    save(output/'release.json',summary)
    body='<aside>Тестовый релиз для локальной проверки. Автоматический выпуск и обработка новых файлов не включены.</aside>'
    body+='<section><h2><a href="call.html">Полный черновик звонка</a></h2><p>Все '+str(review['counts']['speech_runs'])+' сохранённых речевых участков, пометки сомнений, плеер и исходные гипотезы. <a href="call-draft.txt">TXT</a> · <a href="owner-notes.json">Примечания владельца</a></p></section>'
    body+='<section><h2><a href="benchmarks.html">Проверка sound1–4</a></h2><p>Сохранённые результаты и повторная сборка, без новых запусков моделей.</p></section>'
    body+='<section><h2>Что ещё не принято</h2><p>Шумные участки звонка, распознавание пения, разделение детей и взрослых в sound2, изменения слов в sound4. Sound3 — положительный частный результат, не общая гарантия.</p><p>Полнота речи и постоянство собеседников в каналах звонка требуют прослушивания. Времена владельца не использовались для исправления модели.</p></section>'
    failed=[c for c in report['checks'] if not c['replay_pass']]
    body+='<section><h2>Результат аудита</h2><p>Проверено вариантов: '+str(len(report['rows']))+'. Ошибок сборки на уровне записи/серии: '+str(len(failed))+'.</p><p>Итог — пакет для ручной проверки, без разрешения на production.</p></section>'
    (output/'index.html').write_text(document('Poslovnica: тестовый релиз для проверки',body),encoding='utf-8')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quality-root',required=True,type=Path)
    parser.add_argument('--runtime',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    sys.addaudithook(deny_network)
    summary=build(args.quality_root,args.runtime,args.output)
    print(json.dumps({k:v for k,v in summary.items() if k!='input_sha256'},ensure_ascii=False))
