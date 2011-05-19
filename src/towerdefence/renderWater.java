package towerdefence;

import org.newdawn.slick.Animation;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.Image;
import org.newdawn.slick.SlickException;
import org.newdawn.slick.SpriteSheet;
import org.newdawn.slick.state.StateBasedGame;

/**
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class RenderWater {

    SpriteSheet waterSheet;
    Image[] waterFrames;
    Animation waterAnimation;

    int tw, th;

    public RenderWater(int tw, int th) throws SlickException {

        this.tw = tw;
        this.th = th;

        Image water = new Image("data/tilesets/water.png");
        waterSheet = new SpriteSheet(water, 32, 32);
        waterFrames = new Image[8];
        setWaterLeft();

    }

    public void setWaterLeft() {
        for(int x=0;x<waterFrames.length;x++) {
            waterFrames[x]=waterSheet.getSprite(x*3, 1);
        }
        waterAnimation = new Animation(waterFrames, 100,true);
    }

    public void render(GameContainer container, StateBasedGame game, Graphics g) throws SlickException {
        for(int i=0;i<tw;i++) {
            for(int j=0;j<th;j++) {
                waterAnimation.draw(i*32,j*32);
            }
        }

    }

}
