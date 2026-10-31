package ultron;
import java.util.Arrays;
public class GenExp {
  public static void main(String[] a) throws Exception {
    String vocab=a[0], model=a[1], prompt=a[2]; int max=a.length>3?Integer.parseInt(a[3]):80;
    float temp=a.length>4?Float.parseFloat(a[4]):0.7f; int topk=a.length>5?Integer.parseInt(a[5]):40;
    long seed=a.length>6?Long.parseLong(a[6]):1;
    Tokenizer tok=Tokenizer.load(vocab);
    Ultron m; try(java.io.InputStream is=new java.io.FileInputStream(model)){m=Ultron.load(is);}
    if(m.V!=tok.V) throw new RuntimeException("vocab mismatch "+m.V+" "+tok.V);
    Ultron.KVCache c=m.newCache(); Sampler s=new Sampler(m.V,seed);
    int[] ctx=tok.encode(prompt); int shown=ctx.length; float[] logits=null;
    for(int id:ctx) logits=m.step(id,c);
    int EOS=tok.stoi.get(Tokenizer.EOS), PAR=tok.stoi.get(Tokenizer.PAR);
    for(int n=0;n<max;n++){int nx=s.sample(logits,m.V,temp,topk); if(nx==EOS||nx==PAR)break; ctx=Arrays.copyOf(ctx,ctx.length+1);ctx[ctx.length-1]=nx; logits=m.step(nx,c);}
    String out=tok.decode(ctx,shown,ctx.length);
    System.out.println(prompt+out);
  }
}
