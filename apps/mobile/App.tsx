import { useCallback, useEffect, useMemo, useState } from "react";
import { Alert, AppState, Linking, Pressable, ScrollView, StyleSheet, Switch, Text, TextInput, View } from "react-native";
import * as DocumentPicker from "expo-document-picker";
import * as ImagePicker from "expo-image-picker";
import * as SecureStore from "expo-secure-store";
import { File, Paths } from "expo-file-system";
import * as Sharing from "expo-sharing";
import { VideoView, useVideoPlayer } from "expo-video";
import { SafeAreaProvider, SafeAreaView } from "react-native-safe-area-context";
import { SurgicalApiClient } from "@surgical/api-client";
import type { Artifact, Defaults, Job, JobOptions, ModelStatus, PrivacyMode } from "@surgical/api-client";
import { StatusAnimation } from "./src/StatusAnimation";

const TOKEN_KEY = "surgical-api-token";
const ADDRESS_KEY = "surgical-api-address";
const terminal = new Set(["completed", "failed", "cancelled"]);
const defaultOptions: JobOptions = { privacy_mode: "skeleton-only", depth_enabled: true, health_confidence: 0.35, instrument_confidence: 0.25, pose_confidence: 0.25, tracker: "botsort" };

interface SelectedVideo { uri: string; name: string; size: number; duration?: number | null; }

function App() {
  const [address, setAddress] = useState("http://10.0.2.2:8765");
  const [token, setToken] = useState("");
  const [connected, setConnected] = useState(false);
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [defaults, setDefaults] = useState<Defaults | null>(null);
  const [video, setVideo] = useState<SelectedVideo | null>(null);
  const [options, setOptions] = useState<JobOptions>(defaultOptions);
  const [job, setJob] = useState<Job | null>(null);
  const [uploadPercent, setUploadPercent] = useState<number | null>(null);
  const [uploadController, setUploadController] = useState<AbortController | null>(null);
  const [message, setMessage] = useState("Yerel ağdaki analiz bilgisayarına bağlanın.");
  const [page, setPage] = useState<"connection" | "analyze" | "results">("connection");

  const client = useMemo(() => new SurgicalApiClient({ baseUrl: address, token: token || undefined }), [address, token]);

  useEffect(() => {
    void Promise.all([SecureStore.getItemAsync(ADDRESS_KEY), SecureStore.getItemAsync(TOKEN_KEY)]).then(([savedAddress, savedToken]) => {
      if (savedAddress) setAddress(savedAddress);
      if (savedToken) setToken(savedToken);
    });
  }, []);

  const connect = useCallback(async () => {
    try {
      const [health, nextModels, nextDefaults] = await Promise.all([client.checkHealth(), client.listModels(), client.getDefaults()]);
      await SecureStore.setItemAsync(ADDRESS_KEY, address.trim());
      if (token) await SecureStore.setItemAsync(TOKEN_KEY, token); else await SecureStore.deleteItemAsync(TOKEN_KEY);
      setModels(nextModels); setDefaults(nextDefaults); setOptions((current) => ({ ...current, ...nextDefaults }));
      setConnected(true);
      setMessage(health.lan_mode ? "Yerel ağ bağlantısı etkin. TLS olmadan yalnızca güvenilir LAN kullanın." : "Yerel API bağlantısı hazır.");
      setPage("analyze");
    } catch (reason) { setConnected(false); setMessage(reason instanceof Error ? reason.message : "Sunucu bağlantısı kurulamadı."); }
  }, [address, client, token]);

  useEffect(() => {
    if (!job || terminal.has(job.status)) return;
    const poll = async () => { try { setJob(await client.getJob(job.job_id)); } catch { setMessage("İş durumu geçici olarak alınamadı; bağlantı geri geldiğinde tekrar denenecek."); } };
    const subscription = AppState.addEventListener("change", (state) => { if (state === "active") void poll(); });
    const timer = setInterval(() => { if (AppState.currentState === "active") void poll(); }, 1500);
    return () => { subscription.remove(); clearInterval(timer); };
  }, [client, job]);

  async function chooseFile() {
    const result = await DocumentPicker.getDocumentAsync({ type: "video/*", copyToCacheDirectory: false, multiple: false });
    if (!result.canceled) {
      const asset = result.assets[0];
      setVideo({ uri: asset.uri, name: asset.name, size: asset.size ?? 0 });
    }
  }

  async function chooseGallery() {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) { setMessage("Galeri erişimi verilmedi; Dosyalar seçicisini kullanabilirsiniz."); return; }
    const result = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ["videos"], quality: 1, videoExportPreset: ImagePicker.VideoExportPreset.Passthrough });
    if (!result.canceled) {
      const asset = result.assets[0];
      setVideo({ uri: asset.uri, name: asset.fileName ?? "video", size: asset.fileSize ?? 0, duration: asset.duration });
    }
  }

  async function startAnalysis() {
    if (!video) { setMessage("Önce bir video seçin."); return; }
    try {
      const controller = new AbortController();
      setUploadController(controller); setUploadPercent(0); setMessage("Video orijinal kalitesiyle yükleniyor.");
      const file = new File(video.uri);
      const upload = await client.uploadVideo({ data: file as unknown as Blob, filename: video.name, signal: controller.signal, onProgress: (loaded, total) => setUploadPercent(total ? Math.round(loaded / total * 100) : null) });
      const created = await client.createJob({ upload_id: upload.upload_id, options });
      setJob(created); setUploadPercent(null); setUploadController(null); setPage("results"); setMessage("İş kuyruğa alındı.");
    } catch (reason) { setUploadPercent(null); setUploadController(null); setMessage(reason instanceof Error ? reason.message : "Video yüklenemedi."); }
  }

  async function cancelJob() {
    if (job) setJob(await client.cancelJob(job.job_id));
  }

  const skeleton = job?.artifacts.find((item) => item.kind === "skeleton_video");
  const blur = job?.artifacts.find((item) => item.kind === "blur_video");
  return <SafeAreaProvider><SafeAreaView style={styles.safe}><ScrollView contentContainerStyle={styles.screen}>
    <View style={styles.header}><View><Text style={styles.eyebrow}>LOCAL CLIENT</Text><Text style={styles.title}>Surgical Analysis</Text></View><View style={[styles.badge, connected && styles.badgeOk]}><Text style={styles.badgeText}>{connected ? "Bağlı" : "Bağlantı yok"}</Text></View></View>
    <View style={styles.tabs}>{(["connection", "analyze", "results"] as const).map((item) => <Pressable key={item} style={[styles.tab, page === item && styles.tabActive]} onPress={() => setPage(item)}><Text style={styles.tabText}>{item === "connection" ? "Bağlantı" : item === "analyze" ? "Analiz" : "Sonuçlar"}</Text></Pressable>)}</View>
    <View style={styles.notice}><Text style={styles.noticeText}>{message}</Text></View>
    {page === "connection" && <Connection address={address} token={token} connected={connected} models={models} defaults={defaults} onAddress={setAddress} onToken={setToken} onConnect={connect} />}
    {page === "analyze" && <Analyze video={video} options={options} uploadPercent={uploadPercent} onCancelUpload={() => uploadController?.abort()} onFile={chooseFile} onGallery={chooseGallery} onOptions={setOptions} onStart={startAnalysis} disabled={!connected || Boolean(job && !terminal.has(job.status))} />}
    {page === "results" && <Results client={client} job={job} skeleton={skeleton} blur={blur} onCancel={cancelJob} />}
  </ScrollView></SafeAreaView></SafeAreaProvider>;
}

function Connection({ address, token, connected, models, defaults, onAddress, onToken, onConnect }: { address: string; token: string; connected: boolean; models: ModelStatus[]; defaults: Defaults | null; onAddress: (value: string) => void; onToken: (value: string) => void; onConnect: () => Promise<void> }) {
  return <View style={styles.card}><StatusAnimation state={connected ? "idle" : "checking_models"} /><Text style={styles.heading}>Sunucu bağlantısı</Text><Text style={styles.helper}>Android emulator için 10.0.2.2, iOS simulator için 127.0.0.1 kullanılabilir. Fiziksel cihazda bilgisayarın LAN IP adresini girin; değerler otomatik varsayım değildir.</Text><Text style={styles.label}>Sunucu adresi</Text><TextInput style={styles.input} value={address} onChangeText={onAddress} autoCapitalize="none" autoCorrect={false} placeholder="http://192.168.x.x:8765" /><Text style={styles.label}>Token</Text><TextInput style={styles.input} value={token} onChangeText={onToken} secureTextEntry autoCapitalize="none" autoCorrect={false} placeholder="LAN modu için zorunlu" /><Pressable style={styles.primary} onPress={() => void onConnect()}><Text style={styles.primaryText}>Bağlantıyı test et</Text></Pressable><Text style={styles.helper}>Token Expo SecureStore içinde tutulur; normal AsyncStorage, analytics veya telemetry kullanılmaz. TLS olmayan LAN bağlantısında trafiği güvenilir ağla sınırlayın.</Text>{defaults && <Text style={styles.helper}>Varsayılan: {defaults.privacy_mode} · depth {defaults.depth_enabled ? "açık" : "kapalı"}</Text>}<View style={styles.modelGrid}>{models.map((model) => <View key={model.key} style={[styles.model, model.ready ? styles.modelOk : styles.modelWarn]}><Text>{model.key}</Text><Text>{model.ready ? "hazır" : model.state}</Text></View>)}</View></View>;
}

function Analyze({ video, options, uploadPercent, onCancelUpload, onFile, onGallery, onOptions, onStart, disabled }: { video: SelectedVideo | null; options: JobOptions; uploadPercent: number | null; onCancelUpload: () => void; onFile: () => Promise<void>; onGallery: () => Promise<void>; onOptions: (options: JobOptions) => void; onStart: () => Promise<void>; disabled: boolean }) {
  const setMode = (privacy_mode: PrivacyMode) => onOptions({ ...options, privacy_mode });
  return <View style={styles.card}>{uploadPercent !== null && <StatusAnimation state="uploading" />}<Text style={styles.heading}>Video ve analiz ayarları</Text><Text style={styles.helper}>Video telefon üzerinde analiz edilmez; orijinal dosya yeniden kodlanmadan seçili yerel API’ye gönderilir.</Text><View style={styles.row}><Pressable style={styles.secondary} onPress={() => void onFile()}><Text>Dosyalardan seç</Text></Pressable><Pressable style={styles.secondary} onPress={() => void onGallery()}><Text>Galeriden seç</Text></Pressable></View>{video && <View style={styles.file}><Text style={styles.fileName}>{video.name}</Text><Text style={styles.helper}>{(video.size / 1024 / 1024).toFixed(1)} MB{video.duration ? ` · ${(video.duration / 1000).toFixed(1)} sn` : ""}</Text></View>}<Text style={styles.label}>Gizlilik modu</Text><View style={styles.row}>{(["skeleton-only", "blur", "both"] as PrivacyMode[]).map((mode) => <Pressable key={mode} style={[styles.mode, options.privacy_mode === mode && styles.modeActive]} onPress={() => setMode(mode)}><Text>{mode}</Text></Pressable>)}</View>{options.privacy_mode !== "skeleton-only" && <Text style={styles.warning}>Blur/both gerçek görüntüden türetilmiş hassas çıktı üretebilir. Blur video otomatik indirilmez.</Text>}<View style={styles.toggleRow}><Text>Göreli depth üret</Text><Switch value={Boolean(options.depth_enabled)} onValueChange={(depth_enabled) => onOptions({ ...options, depth_enabled })} /></View><View style={styles.row}><View style={styles.half}><Text style={styles.label}>Health confidence</Text><TextInput style={styles.input} keyboardType="decimal-pad" value={String(options.health_confidence ?? 0.35)} onChangeText={(value) => onOptions({ ...options, health_confidence: Number(value) })} /></View><View style={styles.half}><Text style={styles.label}>Alet confidence</Text><TextInput style={styles.input} keyboardType="decimal-pad" value={String(options.instrument_confidence ?? 0.25)} onChangeText={(value) => onOptions({ ...options, instrument_confidence: Number(value) })} /></View></View>{uploadPercent !== null && <View style={styles.row}><Text style={styles.progressText}>Yükleme: %{uploadPercent}</Text><Pressable style={styles.download} onPress={onCancelUpload}><Text style={styles.downloadText}>Yüklemeyi iptal et</Text></Pressable></View>}<Pressable style={[styles.primary, disabled && styles.disabled]} disabled={disabled} onPress={() => void onStart()}><Text style={styles.primaryText}>{uploadPercent !== null ? "Yükleniyor" : "Analizi başlat"}</Text></Pressable></View>;
}

function Results({ client, job, skeleton, blur, onCancel }: { client: SurgicalApiClient; job: Job | null; skeleton?: Artifact; blur?: Artifact; onCancel: () => Promise<void> }) {
  if (!job) return <View style={styles.card}><Text style={styles.heading}>Henüz analiz yok</Text><Text style={styles.helper}>Bir video yükleyip analiz başlattığınızda iş durumu ve artifact dosyaları burada görünür.</Text></View>;
  return <View style={styles.card}><StatusAnimation state={job.status === "completed" ? "completed" : job.status === "failed" ? "error" : "analyzing"} /><Text style={styles.heading}>{job.current_stage}</Text><Text style={styles.helper}>{job.message}</Text><View style={styles.progressTrack}><View style={[styles.progressFill, { width: `${job.progress_percent}%` }]} /></View><Text style={styles.progressText}>%{job.progress_percent} · {job.processed_frames}/{job.total_frames} kare · kalan {job.estimated_remaining_seconds === null ? "—" : `${Math.round(job.estimated_remaining_seconds)} sn`}</Text>{!terminal.has(job.status) && <Pressable style={styles.danger} onPress={() => void onCancel()}><Text style={styles.dangerText}>Güvenli iptal</Text></Pressable>}{job.safe_error_message && <Text style={styles.warning}>{job.safe_error_message}</Text>}{skeleton && <VideoArtifact client={client} jobId={job.job_id} artifact={skeleton} title="Skeleton Video" />}{blur && <View style={styles.sensitive}><Text style={styles.warning}>Blur video hassas içerik olabilir; yalnızca özellikle istenirse indirin.</Text><ArtifactButton client={client} jobId={job.job_id} artifact={blur} confirmSensitive /></View>}<Text style={styles.heading}>Dosyalar</Text>{job.artifacts.map((artifact) => <View style={styles.artifact} key={artifact.artifact_id}><View><Text style={styles.fileName}>{artifact.filename}</Text><Text style={styles.helper}>{artifact.kind} · {(artifact.size_bytes / 1024 / 1024).toFixed(1)} MB</Text></View><ArtifactButton client={client} jobId={job.job_id} artifact={artifact} confirmSensitive={artifact.sensitive} /></View>)}<Text style={styles.helper}>Göreli 4B HTML, metriğe dönüştürülmüş gerçek dünya koordinatı değildir. Paylaşımda blur içerik ayrıca onay ister.</Text></View>;
}

function VideoArtifact({ client, jobId, artifact, title }: { client: SurgicalApiClient; jobId: string; artifact: Artifact; title: string }) {
  const [uri, setUri] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const player = useVideoPlayer(uri, (instance) => { instance.loop = false; });
  async function load() {
    try { setLoading(true); const blob = await client.downloadArtifact(jobId, artifact.artifact_id); const target = new File(Paths.cache, `play_${artifact.filename}`); target.write(new Uint8Array(await blob.arrayBuffer())); setUri(target.uri); } finally { setLoading(false); }
  }
  return <View style={styles.videoGroup}><Text style={styles.heading}>{title}</Text><Text style={styles.helper}>Skeleton-only çıktı daha düşük gizlilik risklidir; tam anonimlik garantisi değildir.</Text>{uri ? <VideoView style={styles.video} player={player} nativeControls /> : <Pressable style={styles.download} onPress={() => void load()}><Text style={styles.downloadText}>{loading ? "Hazırlanıyor" : "Uygulama içinde oynat"}</Text></Pressable>}</View>;
}

function ArtifactButton({ client, jobId, artifact, confirmSensitive }: { client: SurgicalApiClient; jobId: string; artifact: Artifact; confirmSensitive?: boolean }) {
  async function download() {
    if (confirmSensitive) {
      const proceed = await new Promise<boolean>((resolve) => Alert.alert("Hassas çıktı", "Blur çıktısı hassas görüntü içerebilir. Devam edilsin mi?", [{ text: "Vazgeç", style: "cancel", onPress: () => resolve(false) }, { text: "İndir", style: "destructive", onPress: () => resolve(true) }]));
      if (!proceed) return;
    }
    const blob = await client.downloadArtifact(jobId, artifact.artifact_id, Boolean(confirmSensitive));
    const target = new File(Paths.cache, artifact.filename);
    target.write(new Uint8Array(await blob.arrayBuffer()));
    if (await Sharing.isAvailableAsync()) await Sharing.shareAsync(target.uri);
    else await Linking.openURL(target.uri);
  }
  return <Pressable style={styles.download} onPress={() => void download()}><Text style={styles.downloadText}>İndir</Text></Pressable>;
}

const styles = StyleSheet.create({
  safe:{flex:1,backgroundColor:"#eff5f6"},screen:{padding:18,gap:14},header:{flexDirection:"row",justifyContent:"space-between",alignItems:"center",marginBottom:4},eyebrow:{fontSize:11,fontWeight:"700",letterSpacing:1.3,color:"#247b8d"},title:{fontSize:28,fontWeight:"700",color:"#142735"},badge:{paddingHorizontal:10,paddingVertical:6,borderRadius:99,backgroundColor:"#e1e8ea"},badgeOk:{backgroundColor:"#d6f4ed"},badgeText:{fontSize:12,fontWeight:"700",color:"#125668"},tabs:{flexDirection:"row",gap:6},tab:{flex:1,padding:10,borderRadius:8,alignItems:"center",backgroundColor:"#e3edef"},tabActive:{backgroundColor:"#c8eef1"},tabText:{fontWeight:"600",fontSize:12,color:"#185465"},notice:{backgroundColor:"#fff8df",borderRadius:9,padding:11},noticeText:{fontSize:13,color:"#6f520d"},card:{backgroundColor:"#fff",borderRadius:14,padding:17,gap:11,borderWidth:1,borderColor:"#dce9eb"},heading:{fontSize:18,fontWeight:"700",color:"#163039"},helper:{fontSize:12,lineHeight:18,color:"#55717a"},label:{fontSize:12,fontWeight:"700",color:"#42616b",marginTop:3},input:{borderWidth:1,borderColor:"#b8ced2",borderRadius:8,padding:10,fontSize:15,backgroundColor:"#fff"},primary:{backgroundColor:"#087d90",borderRadius:9,padding:13,alignItems:"center",marginTop:5},primaryText:{color:"#fff",fontWeight:"700"},secondary:{backgroundColor:"#eff8f9",borderWidth:1,borderColor:"#bad7dc",borderRadius:8,padding:11,flex:1,alignItems:"center"},row:{flexDirection:"row",gap:9,alignItems:"center"},half:{flex:1},file:{backgroundColor:"#f4fbfb",borderRadius:8,padding:11},fileName:{fontWeight:"700",color:"#17333c",maxWidth:235},mode:{borderWidth:1,borderColor:"#b8ced2",borderRadius:8,padding:9,flex:1,alignItems:"center"},modeActive:{backgroundColor:"#d7f3f4",borderColor:"#1590a3"},warning:{backgroundColor:"#fff8df",color:"#76520b",borderRadius:8,padding:10,fontSize:12,lineHeight:17},toggleRow:{flexDirection:"row",justifyContent:"space-between",alignItems:"center",paddingVertical:6},progressText:{fontSize:12,color:"#42616b"},disabled:{opacity:.5},modelGrid:{flexDirection:"row",flexWrap:"wrap",gap:8},model:{padding:8,borderRadius:8,gap:2,minWidth:100},modelOk:{backgroundColor:"#d6f4ed"},modelWarn:{backgroundColor:"#fff0cf"},progressTrack:{height:10,backgroundColor:"#e0ebed",borderRadius:9,overflow:"hidden"},progressFill:{height:"100%",backgroundColor:"#0d95a8"},danger:{backgroundColor:"#fff0ef",borderColor:"#ffc9c3",borderWidth:1,borderRadius:8,padding:11,alignItems:"center"},dangerText:{color:"#9a3930",fontWeight:"700"},artifact:{borderTopWidth:1,borderTopColor:"#e5eeee",paddingTop:10,flexDirection:"row",justifyContent:"space-between",alignItems:"center",gap:10},download:{backgroundColor:"#eff8f9",borderRadius:8,padding:9},downloadText:{color:"#07596b",fontWeight:"700",fontSize:12},sensitive:{gap:8},videoGroup:{gap:9,marginTop:6},video:{width:"100%",height:220,borderRadius:10,backgroundColor:"#06171a"}
});

export default App;
